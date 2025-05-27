import zipfile
import os
import gzip
import xml.etree.ElementTree as ET
import wget
from lxml import etree
import pandas as pd
import pickle

def unzip_file(zip_path, extract_to):
    # 确保解压目标目录存在
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # 打开ZIP文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 解压所有文件到指定目录
        zip_ref.extractall()
        print(f"Files extracted to: {extract_to}")




dataset = []

base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/XML/"
base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Description/"
csv_base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/"

begin = 1
end = 1000
begin = 497001
end = 498000
while begin < 1973001:
    try:
        begin_str = str(begin).zfill(7)
        end_str = str(end).zfill(7)
        zip_path = begin_str + "_" + end_str + ".zip"
        try:
            file = wget.download(base_url+zip_path)
        except:
            begin += 1000
            end += 1000
            continue
        zip_path = begin_str + "_" + end_str + ".zip"
        extract_to = zip_path.split(".")[0] + "/"
        unzip_file(zip_path, extract_to)
        try:
            file = wget.download(csv_base_url + zip_path)
        except:
            begin += 1000
            end += 1000
            continue
        zip_path = begin_str + "_" + end_str + " (1).zip"

        unzip_file(zip_path, extract_to)
        xml_files = [f for f in os.listdir(extract_to) if f.endswith('.xml.gz')]
        for entry in xml_files:
            # if entry.split(".")[0] == "497001":
            #     print(1)
            full_path = os.path.join(extract_to, entry)
            with gzip.open(full_path, 'rb') as f:
                xml_data = f.read()
            # root = ET.fromstring(xml_data)
            root = etree.fromstring(xml_data)
            namespaces = {'ns': 'http://www.ncbi.nlm.nih.gov'}

            # 提取描述信息
            descriptions = root.findall('.//ns:PC-AssayDescription_description_E', namespaces)
            description_text = ""
            for desc in descriptions:
                if desc.text:
                    description_text += desc.text
                    description_text += " "

            # 提取协议信息
            protocols = root.findall('.//ns:PC-AssayDescription_protocol_E', namespaces)
            protocol_text = ""
            for protocol in protocols:
                if protocol.text:
                    protocol_text += protocol.text
                    protocol_text += " "

            # 提取评论信息
            comments = root.findall('.//ns:PC-AssayDescription_comment_E', namespaces)
            comments_text = ""
            for comment in comments:
                if comment.text:
                    comments_text += comment.text
                    comments_text += " "
            
            full_path = os.path.join(extract_to, entry.split(".")[0]+".csv.gz")
            try:
                with gzip.open(full_path, 'rt') as file:  # 'rt'模式表示读取文本模式
                    data = pd.read_csv(file)
            except:
                continue

            # 筛选出PUBCHEM_ACTIVITY_OUTCOME为'active'的行
            active_data = data[data['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active']
            if active_data.shape[0] == 0:
                continue

            # 提取指定的列
            selected_columns = active_data[['PUBCHEM_SID', 'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES']]
            active_molecules = {}
            active_molecules['SID'] = list(selected_columns['PUBCHEM_SID'])
            active_molecules['CID'] = list(selected_columns['PUBCHEM_CID'])
            active_molecules['SMILES'] = list(selected_columns['PUBCHEM_EXT_DATASOURCE_SMILES'])
            
            # # Get all elements from PC-AssayResults
            # all_results = root.findall(".//ns:PC-AssayResults", namespaces)

            # # Extract active one
            # active_sids = []
            # for result in all_results:
            #     outcome = result.find(".//ns:PC-AssayResults_outcome", namespaces)
            #     if outcome is not None and outcome.get('value') == 'active':
            #         sid = result.find(".//ns:PC-AssayResults_sid", namespaces)
            #         if sid is not None:
            #             active_sids.append(sid.text)
            # if len(active_sids) == 0:
            #     continue

            text = "Description: " + description_text + " Protocol: " + protocol_text + " Comment: " + comments_text
            data = {}
            data['AID'] = entry.split(".")[0]
            data['description'] = text
            data['molecule'] = active_molecules
            dataset.append(data)
    except:
        begin += 1000
        end += 1000
        continue

    begin += 1000
    end += 1000

with open("bioassay_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)