import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
import json
import wget
import pandas as pd
import re
import os
from rdkit import Chem
import numpy as np
import random



num_mol = 8
num_assays = 10
max_molecule_size = 45
random_seed = 42
random.seed(random_seed)


pain_defs = open("CrossDock/pains.txt", "r")
sub_structs = [ line.rstrip().split(" ") for line in pain_defs]
smarts = [ line[0] for line in sub_structs]
pains_desc = [ line[1] for line in sub_structs]
pains_dict = dict(zip(smarts, pains_desc))

def pains_flags_from_smi( smi ):
    try:
        mol = Chem.MolFromSmiles( smi )
        for k,v in pains_dict.items():
            subs = Chem.MolFromSmarts( k )
            if subs != None:
                if mol.HasSubstructMatch( subs ):
                    mol.SetProp(v,k)
        props = [ prop for prop in mol.GetPropNames() ]
        if len(props) == 0:
            props = False
    except:
        props = False 
        pass
    return props


def count_total_atoms_from_mol(mol):
    total_count = mol.GetNumAtoms() if mol else 0
    return total_count

def search_bioassay(query: str, uniprot_id, minimum_example: int = 16,  maximum_BioAssay: int = 8):
    """
    Given a textual protein description as a query, perform a similarity search of BioAssay text using the vectorstore
    and return relevant BioAssays
    Args:
        query: a textual protein description
        uniprot_id: the UniProt ID that corresponds to the protein in the query, which is only used to exclude BioAssays
        corresponding to that protein from the returned list of context
        minimum_example: the minimum number of activity results and SMILES in each BioAssay
        maximum_BioAssay: the maximum number of BioAssays to return

    Returns: relevant BioAssays
    """
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

# Load vectorstore of BioAssay
vectorstore = FAISS.load_local("BioAssay_vectorbase", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

llm = ChatOpenAI(model="chatgpt-4o-latest", openai_api_key="")

# Load CrossDock dataset
with open("CrossDock/CrossDock_test.pkl", "rb") as f:
    crossdock_test = pickle.load(f)

test_mapping = pd.read_csv("CrossDock/test_idmapping_2024_07_09.tsv", sep="\t")
pdb2uniprot = {}
for i in range(test_mapping.shape[0]):
    if test_mapping.iloc[i]['From'] not in list(pdb2uniprot.keys()):
        pdb2uniprot[test_mapping.iloc[i]['From']] = [test_mapping.iloc[i]['Entry']]
    else:
        pdb2uniprot[test_mapping.iloc[i]['From']].append(test_mapping.iloc[i]['Entry'])



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


generate_prompt_base = """
# Role: AI Molecular Generator and BioAssay Analyst

## Profile
- **Author**: LangGPT
- **Version**: 1.1
- **Language**: English
- **Description**: An AI model specialized in analyzing BioAssay results, understanding protein-ligand interactions, and generating high-affinity molecules based on experimental data.

## Skills
- Understanding **protein-ligand interactions** from experimental BioAssay data.
- Interpreting **BioAssay results** and extracting meaningful insights.
- Learning from **high-affinity molecules** in BioAssay data to generate new molecules.
- Ensuring **high binding affinity and specificity**, while avoiding **Pan-assay interference compounds (PAINS)**.
- Generating drug-like molecules that align with known **active reference compounds**.

## Rules
1. **Carefully analyze** the input description of the **protein**, **BioAssay**, and **experimental results**.
2. Identify **high-affinity molecules** (low IC50/Kd values) from the **BioAssay data** as **reference molecules**.
3. Use reference molecules to **learn key functional groups and molecular scaffolds**.
4. Focus on **specificity rather than only high docking scores**.
5. **Each generated molecule should be enclosed within [BOS] and [EOS]**.
6. **Each SMILES should be numbered from 1 to 10, with one per line.**
7. Avoid **PAINS compounds** and prioritize **drug-likeness**.
8. **Do not blindly maximize molecular size**, as larger molecules may have artificially high docking scores but poor specificity.

## Workflows

### **Step 1: Understand the BioAssay and Its Relation to the Query Protein**
- The BioAssays may or may not be related to the **Query Protein**, please identify the correct Query Protein first.
- Carefully interpret the **BioAssay setup**, including:
  - The type of **assay method** used (e.g., enzymatic, fluorescence, cell-based).
  - How the **assay measures protein-ligand interaction**.
  - The **affinity measurements** (e.g., IC50, Kd, Ki).
- Extract **key active molecules** from BioAssay results. (e.g. IC50, Ki, Kd<100nM)
- Identify molecular features that contribute to **high binding affinity**.

### **Step 2: Learn from Active Molecules and Think Step by Step**
- Extract **key functional groups** and **molecular scaffolds** from high-affinity reference molecules.
- Avoid **PAINS compounds** and prioritize **specificity**.
- Ensure molecules remain within a **reasonable drug-like chemical space**.
- Optimize molecular properties for **binding affinity and selectivity**.

### **Step 3: Generate 10 High-Affinity Molecules**
- Use the **active reference molecules** as a learning guide, and use the **low binding affinity molecules** as negative samples.
- Each generated molecule should be optimized for **binding affinity and specificity**.
- The output format must follow this structure:
  - Each **SMILES string should be enclosed in [BOS] and [EOS]**.
  - Each SMILES should be **numbered from 1 to 10**, with each on a separate line.
  - Avoid **PAINS compounds** and prioritize **drug-likeness**.
  - Avoid generating molecules that are too large

### **Step 4: Justify the Molecular Selection**
- Explain how the **reference molecules** influenced the molecular design.
- Describe how the **assay results** guided molecular modifications.
- Justify why these molecules should have **high binding affinity and specificity**.

---

## Output Format:

### **1. BioAssay Understanding & Analysis**
- Step-by-step reasoning about the BioAssay, its setup, and its relevance to the query protein

### **2. Selected Reference Molecules from BioAssay**
- List of highly active molecules from BioAssay used as reference.

### **3. Generated Molecules**

[BOS] SMILES_1 [EOS]
[BOS] SMILES_2 [EOS]
[BOS] SMILES_3 [EOS]
[BOS] SMILES_4 [EOS]
[BOS] SMILES_5 [EOS]
[BOS] SMILES_6 [EOS]
[BOS] SMILES_7 [EOS]
[BOS] SMILES_8 [EOS]
[BOS] SMILES_9 [EOS]
[BOS] SMILES_10 [EOS]

### **4. Justification for Molecular Selection**

- Explanation of how reference molecules influenced design choices, ensuring specificity and affinity while avoiding PAINS.
""" +  "## Query Protein: \n"

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


responses = [[] for i in range(100)]
generated_SMILES = [[] for i in range(100)]
contents = [[] for i in range(100)]
retrieve_assays = []


# Currently only the 'text' and 'pdb_id' fields are used in the CrossDock data
for index in range(100):
    generate_prompt = generate_prompt_base
    protein_description = crossdock_test[index]['text'] + "."
    generate_prompt = generate_prompt + protein_description + "\n## BioAssays"
    query = protein_description

    searched_assays = search_bioassay(query=query, uniprot_id=pdb2uniprot[crossdock_test[index]['pdb_id']], minimum_example=2*num_mol, maximum_BioAssay=num_assays)
    retrieve_assays.append(searched_assays)

    jq_schema = ".PC_AssaySubmit.assay"
    context = ""
    for doc in searched_assays:
        assay = json.loads(doc.page_content)['descr']
        AID = str(assay['aid']['id'])
        summary_input = f"{summary_prompt} \n {protein_description} \n ## **BioAssay JSON ** \n {doc.page_content}"
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
                # Exclude PAINS compounds
                if pains_flags_from_smi(filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"]):
                    continue
                # Control the max_molecule_size
                if count_total_atoms_from_mol(Chem.MolFromSmiles(filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"])) > max_molecule_size:
                    continue

                context = context + filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"] + " " + filtered_df.iloc[i]["PUBCHEM_ACTIVITY_OUTCOME"] + " " + filtered_df.iloc[i]["Standard Type"] + filtered_df.iloc[i]["Standard Relation"] + filtered_df.iloc[i]['Standard Value']
                if not no_unit:
                    context = context + filtered_df.iloc[i]["Standard Units"] +" \n "
                else:
                    context = context + " \n "
            except:
                context = context + filtered_df.iloc[i]["PUBCHEM_EXT_DATASOURCE_SMILES"] + " " + filtered_df.iloc[i]["PUBCHEM_ACTIVITY_OUTCOME"] + " \n "
        
        assay_content = assay_content + "\n ### Randomly Selected Test Compounds \n" + context

        generate_prompt = generate_prompt + "\n" + assay_content
        contents[index] = generate_prompt

    gpt_generated_SMILES = []
    while len(list(set(gpt_generated_SMILES))) < 100:
        response = llm.invoke(generate_prompt)

        responses[index].append(response.content)

        pattern = r"\[BOS\](.*?)\[EOS\]"

        try:
            SMILES = re.findall(pattern, response.content)
            SMILES = list(set([i.strip() for i in SMILES if "and" not in i]))
        except:
            SMILES = "None"
        for smi in SMILES:
            try:
                if Chem.MolFromSmiles(smi) == None:
                    SMILES.remove(smi)
            except:
                SMILES.remove(smi)
        gpt_generated_SMILES.extend(SMILES)
    
    gpt_generated_SMILES = list(set(gpt_generated_SMILES))
        
    generated_SMILES[index].extend(gpt_generated_SMILES)
    with open(f"CrossDock/gpt_input_content_gpt4o.pkl", "wb") as f:
        pickle.dump(contents, f)
    with open(f"CrossDock/gpt_output_content_gpt4o.pkl", "wb") as f:
        pickle.dump(responses, f)
    with open(f"CrossDock/gpt_generated_SMILES_gpt4o.pkl", "wb") as f:
        pickle.dump(generated_SMILES, f)
    with open(f"CrossDock/gpt_retrieve_assay_gpt4o.pkl", "wb") as f:
        pickle.dump(retrieve_assays, f)