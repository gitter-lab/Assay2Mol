import time
import pickle
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.document_loaders import JSONLoader
from langchain.schema import HumanMessage
import json
import os
import pandas as pd
import wget
import random


num_mol = 8
num_assays = 10
max_molecule_size = 45
random_seed = 42
random.seed(random_seed)

vectorstore = FAISS.load_local("BioAssay_vectorbase", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def search_bioassay(query: str, uniprot_id, minimum_example=16,  maximum_BioAssay=8):
    docs_and_score = vectorstore.similarity_search_with_relevance_scores(query, k=1000)
    docs = [i[0] for i in docs_and_score]
    print(f"Number of documents retrieved: {len(docs)}")
    new_docs = []
    count = -1
    for doc in docs:
        count += 1
        try:
            protein_id = [i['mol_id']['protein_accession'] for i in json.loads(doc.page_content)['descr']['target']]
            # If there is overlap between ID, skip this BioAssay as it is directly related to the query protein
            if len(set(protein_id) & set(uniprot_id)) != 0:
                continue
        except:
            pass
        try:
            AID = str(json.loads(doc.page_content)['descr']['aid']['id'])
        except:
            continue
        file = "BioAssay_csv/"+AID+".csv"
        if not os.path.exists(file):
            try:
                file = wget.download("https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+AID+"/CSV", out="BioAssay_csv/"+AID+".csv")
            except:
                continue
        df = pd.read_csv(file)
        df = df[pd.to_numeric(df['PUBCHEM_RESULT_TAG'], errors='coerce').notna()].reset_index(drop=True)
        df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES'])
        if df.shape[0] < minimum_example:
            continue
        new_docs.append(doc)
        if len(new_docs) >= maximum_BioAssay:
            break

    return new_docs


def fill_assay_template(assay, standard_type, summary_of_observations):
    assay = json.loads(assay.page_content)['descr']
    try:
        target_name = ",".join([i['name'] + " cell line" if "is a specific cell-line" in i['descr'] else i['name'] for i in assay['target']])
    except:
        target_name = "None"
    output = """
### ** """ + assay['name'] + """**
- **Target Protein:** """ + target_name  + """
- **Measurement Method:** """ + standard_type + """
- **Summary of Observations:** """ + summary_of_observations + "\n"
    return output


with open("CrossDock/CrossDock_test.pkl", "rb") as f:
    crossdock_test = pickle.load(f)

with open("CrossDock/gpt_generated_SMILES_gpt4o.pkl", "rb") as f:
    initial_SMILES = pickle.load(f)

optimized_SMILES = [[] for i in range(100)]
response = [[] for i in range(100)]


pattern = r"\[BOS\](.*?)\[EOS\]"

llm = ChatOpenAI(model="chatgpt-4o-latest") # , openai_api_key=""



summary_prompt = """

## **Instruction:**
You are an expert in **BioAssay analysis** and **data extraction**. Your task is to carefully analyze the provided BioAssay JSON data and extract structured key information, including:

1. **BioAssay Summarization** – A concise summary of what this assay measures and its scientific purpose.
2. **Assay Type** – The experimental technique used (e.g., **Enzymatic Inhibition, Fluorescence Assay, SPR, Radioligand Binding**).
3. **Summary of Observations** – Important scientific insights derived from the BioAssay, including key patterns in activity, structural features affecting activity, and notable findings.

### **Step-by-step extraction process:**
- Parse the **"descr"** section of the JSON, identifying key information about the assay.
- Identify the **Assay Type** by analyzing the **"name"** field.
- Compare the **Target Protein** found in `"target"` with the one found in `"name"` and `"description"`.  
  - If they **match or are highly related**, set `"Usability_Flag": "Valid"`.  
  - If they are **inconsistent (e.g., different proteins mentioned in different sections)**, set `"Usability_Flag": "Invalid_Target"`.  
- Decide whether the protein studied in the BioAssay is related to **Query Protein**, use protein name, function description and other keywords for your decision.
- Extract **scientific insights** from the description and comments to create the **Summary of Observations**.
- Generate a **concise and informative summary** of the BioAssay, keeping scientific accuracy and relevance.

## **Output Format**
Return the extracted data in the following structured format:

```json
{
  "BioAssay_Summary": "A brief but complete summary of what this assay is measuring and why it is important.",
  "Assay_Type": "The experimental method used (e.g., Enzymatic Inhibition, Fluorescence, SPR, etc.)",
  "Summary_of_Observations": "Scientific insights, key findings, and notable trends from the BioAssay.",
  "CounterScreen": "True" if the BioAssay is identified as CounterScreen against Query Protein, else "False",
}
## **Query Protein**
"""

hERG_description = """
This gene encodes a component of a voltage-activated potassium channel found in cardiac muscle, nerve cells, and microglia. Four copies of this protein interact with one copy of the KCNE2 protein to form a functional potassium channel. Mutations in this gene can cause long QT syndrome type 2 (LQT2). Transcript variants encoding distinct isoforms have been identified. Enables several functions, including protein homodimerization activity; scaffold protein binding activity; and voltage-gated potassium channel activity. Involved in several processes, including membrane repolarization during ventricular cardiac muscle cell action potential; potassium ion transmembrane transport; and regulation of potassium ion transmembrane transport. Acts upstream of or within regulation of heart rate by cardiac conduction. Located in cell surface; perinuclear region of cytoplasm; and plasma membrane. Part of inward rectifier potassium channel complex. Implicated in long QT syndrome; long QT syndrome 2; and short QT syndrome. The KCNH2 gene belongs to a large family of genes that provide instructions for making potassium channels. These channels, which transport positively charged atoms (ions) of potassium out of cells, play key roles in a cell's ability to generate and transmit electrical signals. The specific function of a potassium channel depends on its protein components and its location in the body. Channels made with KCNH2 proteins (also known as hERG1) are active in heart (cardiac) muscle. They are involved in recharging the cardiac muscle after each heartbeat to maintain a regular rhythm. The KCNH2 protein is also produced in nerve cells and certain immune cells (microglia) in the brain and spinal cord (central nervous system). The proteins produced from the KCNH2 gene and another gene, KCNE2, interact to form a functional potassium channel. Four alpha subunits, each produced from the KCNH2 gene, form the structure of each channel. One beta subunit, produced from the KCNE2 gene, attaches (binds) to the channel and regulates its activity.
"""

searched_assays = search_bioassay(query=hERG_description, uniprot_id=[], minimum_example=2*num_mol, maximum_BioAssay=num_assays)

hERG_content = ""

for doc in searched_assays:
    assay = json.loads(doc.page_content)['descr']
    AID = str(assay['aid']['id'])
    summary_input = f"{summary_prompt} \n {hERG_description} \n ## **BioAssay JSON ** \n {doc.page_content}"
    summary_output = None
    while summary_output == None:
        summary_response = llm.invoke(summary_input)
        try:
            summary_output = json.loads(summary_response.content.split("```")[1][4:])
        except:
            continue
    
    file = "BioAssay_csv/"+AID+".csv"
    if not os.path.exists(file):
        try:
            file = wget.download("https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/"+AID+"/CSV", out="BioAssay_csv/"+AID+".csv")
        except:
            continue
    df = pd.read_csv(file)
    df = df[pd.to_numeric(df['PUBCHEM_RESULT_TAG'], errors='coerce').notna()].reset_index(drop=True)
    try:
        standard_type = list(df['Standard Type'])[0]
    except:
        standard_type = "None"
    assay_content = fill_assay_template(doc, standard_type, summary_output['Summary_of_Observations'])

    active_data = df[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active']
    AID = int(AID)
    start = (AID - 1) // 1000 * 1000 + 1
    end = start + 999
    folder = f"{str(start).zfill(7)}_{str(end).zfill(7)}"
    
    if "Standard Units" in df.columns:
        no_unit = False
        selected_columns = ['PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME', 'Standard Type', 'Standard Relation', 'Standard Value', "Standard Units"]
    else:
        no_unit = True
        selected_columns = ['PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME', 'Standard Type', 'Standard Relation', 'Standard Value']

    if active_data.shape[0] == 0:
        df_sampled = df.sample(n=min(2*num_mol, df.shape[0]), random_state=random_seed)
    else:
        active_data = active_data.sample(n=min(num_mol, active_data.shape[0]), random_state=random_seed)
        inactive_data = df[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Active']
        inactive_sampled = inactive_data.sample(n=min(inactive_data.shape[0], active_data.shape[0]), random_state=random_seed)
        df_sampled = pd.concat([inactive_sampled, active_data], ignore_index=True)

    try:
        filtered_df = df_sampled[selected_columns]
    except Exception as e:
        print(AID, " has problem", e)
        try:
            filtered_df = df_sampled[['PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME']]
        except Exception as e:
            print("Has error :", e)
            continue
            
    filtered_df = filtered_df.dropna()
    context = ""
    for i in range(filtered_df.shape[0]):
        try:
            context = context + filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"] + " " + filtered_df.iloc[i]["PUBCHEM_ACTIVITY_OUTCOME"] + " " + filtered_df.iloc[i]["Standard Type"] + filtered_df.iloc[i]["Standard Relation"] + filtered_df.iloc[i]['Standard Value']
            if not no_unit:
                context = context + filtered_df.iloc[i]["Standard Units"] +" \n "
            else:
                context = context + " \n "
        except:
            context = context + filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"] + " " + filtered_df.iloc[i]["PUBCHEM_ACTIVITY_OUTCOME"] + " \n "
    
    assay_content = assay_content + "\n ### Randomly Selected Test Compounds \n" + context

    hERG_content = hERG_content + "\n" + assay_content

hERG_prompt = f"""
To enhance molecular specificity and minimize off-target effects, we aim to reduce potential activity against the hERG channel. {hERG_description} \n\n
{hERG_content} \n\n
Given the retrieved BioAssays for the hERG channel and the associated activity data table, identify molecular features commonly associated with low activity as favorable and those associated with high activity as
undesirable. Using this information, optimize the following ten candidate SMILES strings to reduce their likelihood of interacting with the target.
"""


for protein_num in range(100):
    batch_SMILES = []
    for SMILES in initial_SMILES[protein_num]:
        batch_SMILES.append(SMILES)
        if len(batch_SMILES) == 10:
            SMILES_string = "\n".join(batch_SMILES)
            message = hERG_prompt + SMILES_string + "The output should follow the same format: ten optimized SMILES strings, each enclosed in [BOS] and [EOS], with numbering from 1 to 10."
            human_message = HumanMessage(content=message)
            output = llm([human_message]).content

            response[protein_num].append(output)

            for molecule in re.findall(pattern, output):
                if "and" in molecule:
                    continue
                else:
                    generated_mol = molecule.strip()
                    optimized_SMILES[protein_num].append(generated_mol)
            

    with open("hERG/hERG_optimized_SMILES.pkl", "wb") as f:
        pickle.dump(optimized_SMILES, f)
    with open("hERG/hERG_optimized_response.pkl", "wb") as f:
        pickle.dump(response, f)

