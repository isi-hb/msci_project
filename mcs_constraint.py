#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:36:30 2023

@author: isobelhamilton-burns
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdFMCS
import rdkit

pdb_3d = PandasTools.LoadSDF('Documents/Fourth_Year/HGFR/HGFR_sdf.sdf', removeHs=False)

pdb_3d.head(3)

hgfr_pdb_3d = pdb_3d.iloc[:, [5,7,14,15]]
hgfr_pdb_3d.head(3)

def extract_string(row): return row['model_server_params.value'][17] 

hgfr_pdb_3d['identifier_value'] = hgfr_pdb_3d.apply(extract_string, axis=1)

hgfr_pdb_3d.head()

def concatenate_strings(row): return row['model_server_result.entry_id'] + '_' + row['ID'] + '_' + row['identifier_value']

hgfr_pdb_3d['Molecule_ID'] = hgfr_pdb_3d.apply(concatenate_strings, axis=1)


hgfr_pdb_final = hgfr_pdb_3d.iloc[:,[5,3]]
hgfr_pdb_final.head()

hgfr_sp1_pairs = pd.read_csv('hgfr_singleprotein1_ap.csv')
hgfr_sp2_pairs = pd.read_csv('hgfr_singleprotein2_ap.csv')
hgfr_sp3_pairs = pd.read_csv('hgfr_singleprotein3_ap.csv')
hgfr_sp4_pairs = pd.read_csv('hgfr_singleprotein4_ap.csv')

hgfr_cb1_pairs = pd.read_csv('hgfr_cell1_ap.csv')
hgfr_cb2_pairs = pd.read_csv('hgfr_cell2_ap.csv')
hgfr_cb3_pairs = pd.read_csv('hgfr_cell3_ap.csv')
hgfr_cb4_pairs = pd.read_csv('hgfr_cell4_ap.csv')

hgfr_sp1_A = pd.merge(hgfr_sp1_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_sp2_A = pd.merge(hgfr_sp2_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_sp3_A = pd.merge(hgfr_sp3_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_sp4_A = pd.merge(hgfr_sp4_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')

hgfr_cb1_A = pd.merge(hgfr_cb1_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_cb2_A = pd.merge(hgfr_cb2_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_cb3_A = pd.merge(hgfr_cb3_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')
hgfr_cb4_A = pd.merge(hgfr_cb4_pairs, hgfr_pdb_final, left_on='Closest_PDB_A', right_on='Molecule_ID')

hgfr_sp1_B = pd.merge(hgfr_sp1_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_sp2_B = pd.merge(hgfr_sp2_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_sp3_B = pd.merge(hgfr_sp3_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_sp4_B = pd.merge(hgfr_sp4_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')

hgfr_cb1_B = pd.merge(hgfr_cb1_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_cb2_B = pd.merge(hgfr_cb2_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_cb3_B = pd.merge(hgfr_cb3_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')
hgfr_cb4_B = pd.merge(hgfr_cb4_A, hgfr_pdb_final, left_on='Closest_PDB_B', right_on='Molecule_ID')

def smiles_to_mol(smiles): return AllChem.MolFromSmiles(smiles)

hgfr_sp1_B['Mol_chem_A'] = hgfr_sp1_B['Structure_A'].apply(smiles_to_mol)
hgfr_sp2_B['Mol_chem_A'] = hgfr_sp2_B['Structure_A'].apply(smiles_to_mol)
hgfr_sp3_B['Mol_chem_A'] = hgfr_sp3_B['Structure_A'].apply(smiles_to_mol)
hgfr_sp4_B['Mol_chem_A'] = hgfr_sp4_B['Structure_A'].apply(smiles_to_mol)

hgfr_cb1_B['Mol_chem_A'] = hgfr_cb1_B['Structure_A'].apply(smiles_to_mol)
hgfr_cb2_B['Mol_chem_A'] = hgfr_cb2_B['Structure_A'].apply(smiles_to_mol)
hgfr_cb3_B['Mol_chem_A'] = hgfr_cb3_B['Structure_A'].apply(smiles_to_mol)
hgfr_cb4_B['Mol_chem_A'] = hgfr_cb4_B['Structure_A'].apply(smiles_to_mol)


hgfr_sp1_B['Mol_chem_B'] = hgfr_sp1_B['Structure_B'].apply(smiles_to_mol)
hgfr_sp2_B['Mol_chem_B'] = hgfr_sp2_B['Structure_B'].apply(smiles_to_mol)
hgfr_sp3_B['Mol_chem_B'] = hgfr_sp3_B['Structure_B'].apply(smiles_to_mol)
hgfr_sp4_B['Mol_chem_B'] = hgfr_sp4_B['Structure_B'].apply(smiles_to_mol)

hgfr_cb1_B['Mol_chem_B'] = hgfr_cb1_B['Structure_B'].apply(smiles_to_mol)
hgfr_cb2_B['Mol_chem_B'] = hgfr_cb2_B['Structure_B'].apply(smiles_to_mol)
hgfr_cb3_B['Mol_chem_B'] = hgfr_cb3_B['Structure_B'].apply(smiles_to_mol)
hgfr_cb4_B['Mol_chem_B'] = hgfr_cb4_B['Structure_B'].apply(smiles_to_mol)

hgfr_sp1 = hgfr_sp1_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_sp2 = hgfr_sp2_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_sp3 = hgfr_sp3_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_sp4 = hgfr_sp4_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]

hgfr_cb1 = hgfr_cb1_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_cb2 = hgfr_cb2_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_cb3 = hgfr_cb3_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]
hgfr_cb4 = hgfr_cb4_B.iloc[:, [1,2,3,4,5,16,13,6,7,8,9,10,17,15,11]]

hgfr_sp1 = hgfr_sp1.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_sp2 = hgfr_sp2.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_sp3 = hgfr_sp3.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_sp4 = hgfr_sp4.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})

hgfr_cb1 = hgfr_cb1.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_cb2 = hgfr_cb2.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_cb3 = hgfr_cb3.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})
hgfr_cb4 = hgfr_cb4.rename(columns ={'ROMol_x' : 'Mol_PDB_A', 'ROMol_y' : 'Mol_PDB_B'})

hgfr_sp1.head()

hgfr_mcs_total = pd.concat([hgfr_sp1, hgfr_sp2, hgfr_sp3, hgfr_sp4, hgfr_cb1, hgfr_cb2, hgfr_cb3, hgfr_cb4])

#Find the most common compounds to the pairs 

counts = hgfr_mcs_total['ChEMBL_ID_A'].value_counts()
hgfr_top_100 = counts.head(100)

print(hgfr_top_100)

hgfr_100 = hgfr_mcs_total[hgfr_mcs_total['ChEMBL_ID_A'].isin(hgfr_top_100.index)]

print(hgfr_100)

#Finding the unique values in the dataframe - should get rid of all the duplication 
chembl_values = hgfr_100['ChEMBL_ID_A'].tolist()

unique_values = set(chembl_values)
count = len(unique_values)
count

unique_list = list(unique_values)

hgfr_chembl_values = pd.DataFrame(unique_list, columns=['ChEMBL_ID'])

hgfr_mcs_final = pd.merge(hgfr_chembl_values, hgfr_mcs_total, left_on='ChEMBL_ID', right_on='ChEMBL_ID_A')

hgfr_mcs = hgfr_mcs_final.drop_duplicates(subset='ChEMBL_ID', inplace=False)

hgfr_mcs = hgfr_mcs.iloc[:,[1,2,3,4,5,6,7]]

hgfr_mcs.reset_index(drop=True, inplace=True)

print(hgfr_mcs)
hgfr_mcs.head()


#calculate the MCS between the molecule and the PDB structure 
#Add this to the dataframe

def calculate_mcs(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1, mol2])
    smarts = mcs.smartsString
    return AllChem.MolFromSmarts(smarts)

def calculate_mcs_dataframe(row):
    """
    Calculates the maximum common substructure between two molecules in a dataframe row.
    """
    mol1 = row['Mol_chem_A']
    mol2 = row['Mol_PDB_A']
    mcs = calculate_mcs(mol1, mol2)
    return Chem.MolToSmiles(mcs)

#Try with the sample set of 10 - took about 30 seconds 

hgfr_mcs_10['mcs_smarts'] = hgfr_mcs_10.apply(calculate_mcs_dataframe, axis=1)
print(hgfr_mcs_10)

#With sample set of 20 - about 1-2 minutes 

hgfr_mcs_20['mcs_smarts'] = hgfr_mcs_20.apply(calculate_mcs_dataframe, axis=1)
print(hgfr_mcs_20)

#With set of 50 - about 18 minutes (lord knows why)

hgfr_mcs_50['mcs_smarts'] = hgfr_mcs_50.apply(calculate_mcs_dataframe, axis=1)
print(hgfr_mcs_50)

#With of 100 - compare computation time - actually only took 20 minutes so that's a win

hgfr_mcs_100['mcs_smarts'] = hgfr_mcs_100.apply(calculate_mcs_dataframe, axis=1)
print(hgfr_mcs_100)

#using the 'most popular' dataframe with 100 entries - took a little while but worked! 
hgfr_mcs['mcs_smarts'] = hgfr_mcs.apply(calculate_mcs_dataframe, axis=1)
print(hgfr_mcs)


hgfr_mcs.to_csv('hgfr_mcs.csv')


patt = hgfr_mcs.iloc[0,-1]
patt

mcs_patt = [Chem.MolFromSmarts(smarts_string) for smarts_string in hgfr_mcs['mcs_smarts']]
hgfr_mcs['mcs_patt'] = mcs_patt
    

#Create a list of conformers for each chembl molecule in the Dataframe

#Function to generate 10 conformers for each molecule using the ETKDG method
def generate_conformers(mol):
    num_confs = 10
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, num_confs, params)
    conformers = [mol.GetConformer(i) for i in range(num_confs)]
    return conformers

#Applying that function to each molcule and creating a list of conformers
conformers_list = []
for idx, row in hgfr_mcs.iterrows():
    mol = row["Mol_chem_A"]
    conformers = generate_conformers(mol)
    conformers_list.append(conformers)
    print(conformers_list)

#Appending this list to the existing dataframe 
hgfr_mcs["conformers"] = conformers_list
print(hgfr_mcs)



#Find the number of atoms in each of the MCS molecules found 

num_atoms_list = []
for mol in hgfr_mcs["mcs_patt"]:
    num_atoms = mol.GetNumAtoms()
    num_atoms_list.append(num_atoms)
    
hgfr_mcs["mcs_num_atoms"] = num_atoms_list

print(hgfr_mcs)

hgfr_mcs.iloc[0,:]

#Do the alignment with just a single molecule and its PDB 

mol = hgfr_mcs.iloc[0, 5]
ref_mol = hgfr_mcs.iloc[0,6]
mcs = hgfr_mcs.iloc[0,8]
conformer_list = hgfr_mcs.iloc[0,9]
conformer = conformer_list[0]

print(mol)
print(conformer_list)
print(ref_mol)
print(conformer)

mcs = Chem.rdFMCS.FindMCS([ref_mol, mol])

print(mcs)

mcs_smarts = mcs.smartsString

print(mcs_smarts)

atom_map = ref_mol.GetSubstructMatch(Chem.MolFromSmarts(mcs_smarts))

print(atom_map)

atom_map_list = list(atom_map)

print(atom_map_list)


rdMolAlign.AlignMol(mol, ref_mol, atomMap=atom_map_list)

rdkit.Chem.rdMolAlign.AlignMolConformers(ref_mol)



##################
#Try the alignment with a smaller test set of data

patt = hgfr_mcs_10.iloc[0,-1]
patt

mcs_patt = [Chem.MolFromSmarts(smarts_string) for smarts_string in hgfr_mcs_10['mcs_smarts']]
hgfr_mcs_10['mcs_patt'] = mcs_patt
    

#Create a list of conformers for each chembl molecule in the Dataframe

#Function to generate 10 conformers for each molecule using the ETKDG method
def generate_conformers(mol):
    num_confs = 10
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol, num_confs, params)
    conformers = [mol.GetConformer(i) for i in range(num_confs)]
    return conformers

#Applying that function to each molcule and creating a list of conformers
conformers_list = []
for idx, row in hgfr_mcs_10.iterrows():
    mol = row["Mol_chem_A"]
    conformers = generate_conformers(mol)
    conformers_list.append(conformers)
    print(conformers_list)

#Appending this list to the existing dataframe 
hgfr_mcs_10["conformers"] = conformers_list
print(hgfr_mcs_10)



#Find the number of atoms in each of the MCS molecules found 

num_atoms_list = []
for mol in hgfr_mcs_10["mcs_patt"]:
    num_atoms = mol.GetNumAtoms()
    num_atoms_list.append(num_atoms)
    
hgfr_mcs_10["mcs_num_atoms"] = num_atoms_list

print(hgfr_mcs_10)


#Try to align each PDB mol with the 10 conformers 
#Use the MCS as a atom map to constrain that alignment 
#Find the rmsd for this alignment 






