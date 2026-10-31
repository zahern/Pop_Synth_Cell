import csv
import logging
import os
import pandas as pd
import warnings


# ---------------------------------------------------------------------------
# SEQ (South East Queensland) + Tweed Shire geography
# ---------------------------------------------------------------------------
# Previously this was a numeric SA4-prefix range (301-311), which was both
# wrong (it excluded real SEQ SA4s 313/314 Moreton Bay N/S and 316 Sunshine
# Coast, while a separate, much broader ad-hoc filter elsewhere - see
# sa_multi_wrapper.py's _is_study_area_zone - masked the omission by matching
# on ANY Queensland SA2) and incomplete (SEQ under the official ShapingSEQ /
# SEQ Regional Plan boundary also includes Toowoomba SA4 317, and the Gold
# Coast's actual NSW commuting catchment is Tweed Shire only - SA3 11203
# "Tweed Valley" - not the whole Richmond-Tweed SA4 112, which also drags in
# Lismore/Casino/Kyogle/Byron Bay/Ballina/Evans Head).
#
# The authoritative zone list is now the explicit SA2 code set in
# sa2_zones_reference_357.csv (built from the real ABS SA2/SA3/SA4 2021
# boundaries: SA4 301,302,303,304,305 Brisbane, 309 Gold Coast, 310 Ipswich,
# 311 Logan-Beaudesert, 313,314 Moreton Bay N/S, 316 Sunshine Coast,
# 317 Toowoomba, plus SA3 11203 Tweed Valley) - 357 zones, fixed 2026-07.
# Extended 2026-07-16 (+14 zones -> 371 total): SA3 11201 Richmond Valley -
# Coastal and 11202 Richmond Valley - Hinterland, so the SA-GAN control
# geography covers every zone in the ABM's operative 324-zone skims/landuse
# list (zone_ids_sa_only_507.txt) - those NSW zones previously had skims and
# landuse but no synthetic population. Filename keeps the historical "357".
#
# A copy of this file lives HERE (in synth_pop/, this project's natural home
# for zone definitions) and is kept in sync with the copy in the sibling
# test_runs_tours/demand_ project, which is the actual source of truth (built
# directly from the ABS SA2_2021_AUST_GDA2020 shapefile - regenerate from
# there, then re-copy, if the boundary ever changes). The local copy is tried
# first so this project doesn't have a hard runtime dependency on the sibling
# project's directory existing at a fixed relative path.
_SEQ_ZONE_LIST_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sa2_zones_reference_357.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "test_runs_tours", "demand_", "sa2_zones_reference_357.csv"),
]


def _load_seq_sa2_codes() -> frozenset:
    """Load the explicit SEQ+Tweed SA2 code set. Cached after first call."""
    for candidate in _SEQ_ZONE_LIST_CANDIDATES:
        path = os.path.normpath(candidate)
        try:
            with open(path, newline="", encoding="utf-8") as f:
                codes = frozenset(row["sa2_code"].strip() for row in csv.DictReader(f))
            if codes:
                return codes
        except Exception:
            continue
    warnings.warn(
        f"is_seq_sa2: could not load any of {_SEQ_ZONE_LIST_CANDIDATES}; "
        "falling back to the old, less accurate 301-311 SA4-prefix range."
    )
    # Fallback only - known to be wrong (misses 313/314/316/317/Tweed Valley),
    # kept so callers don't hard-crash if the reference file has moved.
    return frozenset()


_SEQ_SA2_CODES = _load_seq_sa2_codes()
# Legacy name kept for any external references; no longer used for filtering.
SEQ_SA4_PREFIXES: tuple = tuple(f"{i:03d}" for i in range(301, 312))


def is_seq_sa2(code) -> bool:
    """Return True if a 9-digit SA2 code is in the SEQ+Tweed Shire study area
    (357-zone list in sa2_zones_reference_357.csv)."""
    s = str(code).strip()
    if len(s) != 9 or not s.isdigit():
        return False
    if _SEQ_SA2_CODES:
        return s in _SEQ_SA2_CODES
    return s[:3] in SEQ_SA4_PREFIXES   # stale fallback - see _load_seq_sa2_codes


# ---------------------------------------------------------------------------


class DataMappings:
    def __init__(self):
        # Initialize all mappings as class attributes
        self._init_mappings()

    
    def _init_mappings(self):
        self.age_map = {
            "0 years": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
            "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
            "10 years": 10, "11 years": 11, "12 years": 12, "13 years": 13,
            "14 years": 14, "15 years": 15, "16 years": 16, "17 years": 17,
            "18 years": 18, "19 years": 19, "20 years": 20, "21 years": 21,
            "22 years": 22, "23 years": 23, "24 years": 24, "25-29 years": 25,
            "30-34 years": 26, "35-39 years": 27, "40-44 years": 28,
            "45-49 years": 29, "50-54 years": 30, "55-59 years": 31,
            "60-64 years": 32, "65-69 years": 33, "70-74 years": 34,
            "75-79 years": 35, "80-84 years": 36, "85 years and over": 37,
        }
        self.age5p_map = {
            "0-4 years": 1,
            "5-9 years": 2,
            "10-14 years": 3,
            "15-19 years": 4,
            "20-24 years": 5,
            "25-29 years": 6,
            "30-34 years": 7,
            "35-39 years": 8,
            "40-44 years": 9,
            "45-49 years": 10,
            "50-54 years": 11,
            "55-59 years": 12,
            "60-64 years": 13,
            "65-69 years": 14,
            "70-74 years": 15,
            "75-79 years": 16,
            "80-84 years": 17,
            "85 years and over": 18,
        }

        self.vehrd_map = {
            "No motor vehicles": 0, "One motor vehicle": 1, "Two motor vehicles": 2,
            "Three motor vehicles": 3, "Four or more motor vehicles": 4,
            # 2026-07-09 fix: "Not stated" and "Not applicable" were both mapped to 5,
            # silently merging two distinct categories into one. GAN training data
            # (LINKED_DATA.csv) keeps them separate (7 distinct VEHRD codes: 0-6),
            # so census-derived targets built from this map were undercounting by
            # one whole category. Split them, following this codebase's consistent
            # ABS convention of "Not applicable" as the last/highest code (see
            # NPRD=7, RNTRD=27, MRERD=23, OCCP=11 -- all "Not applicable" last).
            "Not stated": 5, "Not applicable": 6,
        }

        self.occp_map = {
            "Managers": 1, "Professionals": 2, "Technicians and Trades Workers": 3,
            "Community and Personal Service Workers": 4,
            "Clerical and Administrative Workers": 5, "Sales Workers": 6,
            "Machinery Operators and Drivers": 7, "Labourers": 8,
            "Inadequately described": 9, "Not stated": 10,
            "Not applicable": 11,
        }
        self.rlhp_map = {
            "Husband, Wife or Partner": "1",
            "Lone parent": "2",
            "Child under 15": "3",
            "Dependent student": "4",
            "Non-dependent child": "5",
            "Other related individual": "6",
            "Non-family member": "7",
            "Non-classifiable": "9",
            "Not applicable": "11",
        }
        self.mstp_map = {
        "Never married" : "1",
        "Widowed" : "2",
        "Divorced" : "3",
        "Separated" : "4",
        "Married" : "5",
        "Not applicable" : "6"

        }
        self.sexp_map = {
            "Male": "1",
            "Female": "2"
        }


        # Mapping for person categories
        self.nprd_map = {
            "One person": "1",
            "Two persons": "2",
            "Three persons": "3",
            "Four persons": "4",
            "Five persons": "5",
            "Six persons": "6",
            "Not applicable": "7"
        }

        # Mapping for income ranges
        self.hind_map = {
            "Negative income": 1,
            "Nil income": 2,
            "$1-$149 ($1-$7,799)": 3,
            "$150-$299 ($7,800-$15,599)": 4,
            "$300-$399 ($15,600-$20,799)": 5,
            "$400-$499 ($20,800-$25,999)": 6,
            "$500-$649 ($26,000-$33,799)": 7,
            "$650-$799 ($33,800-$41,599)": 8,
            "$800-$999 ($41,600-$51,999)": 9,
            "$1,000-$1,249 ($52,000-$64,999)": 10,
            "$1,250-$1,499 ($65,000-$77,999)": 11,
            "$1,500-$1,749 ($78,000-$90,999)": 12,
            "$1,750-$1,999 ($91,000-$103,999)": 13,
            "$2,000-$2,499 ($104,000-$129,999)": 14,
            "$2,500-$2,999 ($130,000-$155,999)": 15,
            "$3,000-$3,499 ($156,000-$181,999)": 16,
            "$3,500-$3,999 ($182,000-$207,999)": 17,
            "$4,000-$4,499 ($208,000-$233,999)": 18,
            "$4,500-$4,999 ($234,000-$259,999)": 19,
            "$5,000-$5,999 ($260,000-$311,999)": 20,
            "$6,000-$7,999 ($312,000-$415,999)": 21,
            "$8,000 or more ($416,000 or more)": 22,
            "Partial income stated": 23,
            "All incomes not stated": 24,
            "Not applicable": 25
        }

        # Mapping for mortgage repayments
        self.mrtrd_map = {
            "Nil repayments": 1,
            "$1-$149": 2,
            "$150-$299": 3,
            "$300-$449": 4,
            "$450-$599": 5,
            "$600-$799": 6,
            "$800-$999": 7,
            "$1,000-$1,199": 8,
            "$1,200-$1,399": 9,
            "$1,400-$1,599": 10,
            "$1,600-$1,799": 11,
            "$1,800-$1,999": 12,
            "$2,000-$2,199": 13,
            "$2,200-$2,399": 14,
            "$2,400-$2,599": 15,
            "$2,600-$2,799": 16,
            "$2,800-$2,999": 17,
            "$3,000-$3,499": 18,
            "$3,500-$3,999": 19,
            "$4,000-$4,999": 20,
            "$5,000 and over": 21,
            "Not stated": 22,
            "Not applicable": 23
        }

        # Mapping for rent ranges
        self.rntrd_map = {
            "$1-$74": 1,
            "$75-$99": 2,
            "$100-$124": 3,
            "$125-$149": 4,
            "$150-$174": 5,
            "$175-$199": 6,
            "$200-$224": 7,
            "$225-$249": 8,
            "$250-$274": 9,
            "$275-$299": 10,
            "$300-$324": 11,
            "$325-$349": 12,
            "$350-$374": 13,
            "$375-$399": 14,
            "$400-$424": 15,
            "$425-$449": 16,
            "$450-$474": 17,
            "$475-$499": 18,
            "$500-$524": 19,
            "$525-$549": 20,
            "$550-$649": 21,
            "$650-$749": 22,
            "$750-$849": 23,
            "$850-$949": 24,
            "$950 and over": 25,
            "Not stated": 26,
            "Not applicable": 27
        }

        # Mapping for education levels
        self.heap_map = {
            "Postgraduate Degree Level": 1,
            "Graduate Diploma and Graduate Certificate Level": 2,
            "Bachelor Degree Level": 3,
            "Advanced Diploma and Diploma Level": 4,
            "Certificate III & IV Level": 5,
            "Secondary Education - Years 10 and above": 6,
            "Certificate I & II Level": 7,
            "Secondary Education - Years 9 and below": 8,
            "Supplementary Codes": 9,
            "Not stated": 10,
            "Not applicable": 11,
            "Overseas visitor": 12
        }

        # Mapping for travel modes
        self.mtwp_map = {
            "Train": 1,
            "Bus": 2,
            "Ferry": 3,
            "Tram/light rail": 4,
            "Taxi/ride-share service": 5,
            "Car, as driver": 6,
            "Car, as passenger": 7,
            "Truck": 8,
            "Motorbike/scooter": 9,
            "Bicycle": 10,
            "Walked only": 11,
            "Other Mode": 12,
            "Worked at home": 13,
            "Did not go to work": 14,
            "Not stated": 15,
            "Not applicable": 16,
            "Overseas visitor": 17
        }


        # Add other mappings as attributes (e.g., income_map, rlhp_map, etc.)
    
    def map_column(self, df, column_name, mapping):
        """
        Maps the values of a column in the DataFrame based on a given mapping.

        Parameters:
        - df: The input DataFrame.
        - column_name: The name of the column to map.
        - mapping: The dictionary mapping original values to their mapped values.

        Returns:
        - A DataFrame with the mapped column values.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")
        
        # Apply the mapping
        df[column_name] = df[column_name].map(mapping).fillna(df[column_name])
        return df

    def map_columns(self, df, mappings):
        """
        Maps multiple columns in the DataFrame based on their respective mappings.

        Parameters:
        - df: The input DataFrame.
        - mappings: A dictionary where keys are column names and values are mappings.

        Returns:
        - A DataFrame with the mapped column values.
        """
        for column, mapping in mappings.items():
            df = self.map_column(df, column, mapping)
        return df



class CensusControlTable:
    def __init__(self, file_path, remove_totals=True, simplify_names=True, *args, **kwargs):
        """
        Initialize the CensusControlTable object by reading and processing the control table.

        Parameters:
            file_path (str): Path to the CSV file.
            remove_totals (bool): Whether to remove rows with 'Total'.
            simplify_names (bool): Whether to simplify column names.
        """
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        #self.file_path = file_path
        self.remove_totals = remove_totals
        self.simplify_names = simplify_names
        self.control_table = self._read_and_process_table()
        self.sa_num = kwargs.get('zone_level', 'sa2')
        self.define_mapping()
        
    
        

    def define_mapping(self):
        # Initialize all mappings as class attributes



        self.mappings = {
            
            #CORE ACTIVITY FOR ASSISTANCE
        "assnp":{"Has need for assistance with core activities":1, 
                 "Does not have need for assistance with core activities":2,
                 "Not stated":3, 
                 "Overseas visitor":4},
         
            
        "agep" : {
            "0 years": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
            "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
            "10 years": 10, "11 years": 11, "12 years": 12, "13 years": 13,
            "14 years": 14, "15 years": 15, "16 years": 16, "17 years": 17,
            "18 years": 18, "19 years": 19, "20 years": 20, "21 years": 21,
            "22 years": 22, "23 years": 23, "24 years": 24, "25-29 years": 25,
            "30-34 years": 26, "35-39 years": 27, "40-44 years": 28,
            "45-49 years": 29, "50-54 years": 30, "55-59 years": 31,
            "60-64 years": 32, "65-69 years": 33, "70-74 years": 34,
            "75-79 years": 35, "80-84 years": 36, "85 years and over": 37,
        },
            "gngp":{
        "National Government": 1,
        "State/Territory Government": 2,
        "Local Government": 3,
        "Private sector": 4,
        "Not stated": 5,
        "Not applicable": 6,
        "Overseas visitor": 7
            } ,
                        "spip": {
                "Family reference person": 1,
                "Spouse/partner": 2,
                "Not applicable": 3,
                "Overseas visitor": 4
            },

            "typp": {
                "Preschool": 1,
                "Primary - Government": 2,
                "Primary - Catholic": 3,
                "Primary - other non-Government": 3,
                "Primary - not further defined": 3,
                "Secondary - Government": 4,
                "Secondary - Catholic": 5,
                "Secondary - other non-Government": 5,
                "Secondary - not further defined": 5,
                "Tertiary - Vocational education (including TAFE and private training providers)": 6,
                "Tertiary - University or other higher education": 7,
                "Tertiary - not further defined": 6,
                "Other": 8,
                "Not stated": 9,
                "Not applicable": 10,
                "Overseas visitor": 11
            },

            "indp": {
                "Agriculture, Forestry and Fishing": 1,
                "Mining": 2,
                "Manufacturing": 3,
                "Electricity, Gas, Water and Waste Services": 4,
                "Construction": 5,
                "Wholesale Trade": 6,
                "Retail Trade": 7,
                "Accommodation and Food Services": 8,
                "Transport, Postal and Warehousing": 9,
                "Information Media and Telecommunications": 10,
                "Financial and Insurance Services": 11,
                "Rental, Hiring and Real Estate Services": 12,
                "Professional, Scientific and Technical Services": 13,
                "Administrative and Support Services": 14,
                "Public Administration and Safety": 15,
                "Education and Training": 16,
                "Health Care and Social Assistance": 17,
                "Arts and Recreation Services": 18,
                "Other Services": 19,
                "Inadequately described": 20,
                "Not stated": 21,
                "Not applicable": 22,
                "Overseas visitor": 23
            },



        "age5p": {
            "0-4 years": [0, 1, 2, 3, 4],
            "5-9 years": [5, 6, 7, 8, 9],
            "10-14 years": [10, 11, 12, 13, 14],
            "15-19 years": [15, 16, 17, 18, 19],
            "20-24 years": [20, 21, 22, 23, 24],
            "25-29 years": 25,
            "30-34 years": 26,
            "35-39 years": 27,
            "40-44 years": 28,
            "45-49 years": 29,
            "50-54 years": 30,
            "55-59 years": 31,
            "60-64 years": 32,
            "65-69 years": 33,
            "70-74 years": 34,
            "75-79 years": 35,
            "80-84 years": 36,
            "85 years and over": 37
        },
            "age5HHTS": {
                "0-4 years": [1],
                "5-9 years": [2],
                "10-14 years": [3],
                "15-19 years": [4],
                "20-24 years": [5],
                "25-29 years": 6,
                "30-34 years": 7,
                "35-39 years": 8,
                "40-44 years": 9,
                "45-49 years": 10,
                "50-54 years": 11,
                "55-59 years": 12,
                "60-64 years": 13,
                "65-69 years": 14,
                "70-74 years": 15,
                "75-79 years": 16,
                "80-84 years": 17,
                "85-89 years": 18,
                "90-94 years": 19,
                "95-99 years": 20,
                "100 years and over": 21

            },

        "vehrd": {
            "No motor vehicles": 0, "One motor vehicle": 1, "Two motor vehicles": 2,
            "Three motor vehicles": 3, "Four or more motor vehicles": 4,
            # 2026-07-09 fix: see self.vehrd_map above for why these must be
            # separate codes (GAN training data has 7 distinct VEHRD codes, not 6).
            "Not stated": 5, "Not applicable": 6,
        },

        "occp": {
            "Managers": 1, "Professionals": 2, "Technicians and Trades Workers": 3,
            "Community and Personal Service Workers": 4,
            "Clerical and Administrative Workers": 5, "Sales Workers": 6,
            "Machinery Operators and Drivers": 7, "Labourers": 8,
            "Inadequately described": 9, "Not stated": 10,
            "Not applicable": 11,
        },
        "rlhpk": {
            "Husband, Wife or Partner": 1,
            "Lone parent": 2,
            "Child under 15": 3,
            "Dependent student": 4,
            "Non-dependent child": 5,
            "Other related individual": 6,
            "Non-family member": 7,
            "Non-classifiable": 9,
            "Not applicable": 10,  # First "Not applicable"
            "Overseas": 11  # Overwrites the previous key as keys must be unique
        },
            # Condensed mapping dictionary
        'rlhp_map' : {
            "Husband, Wife or Partner": [1, 2, 3, 4, 5, 6],
            "Lone parent": [7],
            "Child under 15": [8, 9, 10, 11, 12, 13],
            "Dependent student": [14, 15, 16],
            "Non-dependent child": [17, 18, 19],
            "Other related individual": [20, 21, 22, 23, 24, 25, 26, 27],
            "Non-family member": [28, 29, 30, 31],
            "Non-classifiable": [32],
            "Not applicable": [33],  # Example of a potential extra category
            "Overseas": [34]  # Example of a potential extra category
        },
        "relationship":{
            "Husband, Wife or Partner in a registered marriage, opposite-sex couple": 1,
            "Husband, Wife or Partner in a registered marriage, male same-sex couple": 2,
            "Husband, Wife or Partner in a registered marriage, female same-sex couple": 3,
            "Husband, Wife or Partner in de facto marriage, opposite-sex couple": 4,
            "Husband, Wife or Partner in de facto marriage, male same-sex couple": 5,
            "Husband, Wife or Partner in de facto marriage, female same-sex couple": 6,
            "Lone parent": 7,
            "Natural or adopted child under 15": 8,
            "Step child under 15": 9,
            "Foster child under 15": 10,
            "Grandchild under 15": 11,
            "Otherwise related child under 15": 12,
            "Unrelated child under 15": 13,
            "Natural or adopted dependent student": 14,
            "Dependent student step child": 15,
            "Dependent student foster child": 16,
            "Non-dependent natural, or adopted child": 17,
            "Non-dependent step child": 18,
            "Non-dependent foster child": 19,
            "Brother/sister": 20,
            "Father/mother": 21,
            "Non-dependent grandchild": 22,
            "Grandfather/grandmother": 23,
            "Cousin": 24,
            "Uncle/aunt": 25,
            "Nephew/niece": 26,
            "Other related individual (nec)": 27,
            "Unrelated individual living in family household": 28,
            "Group household member": 29,
            "Lone person": 30,
            "Visitor (from within Australia)": 31,
            "Other non-classifiable relationship": 32,
            "Not applicable": 33,
            "Overseas visitor": 34
        },

        "rlhp": {
                "Husband, Wife or Partner in a registered marriage, opposite-sex couple": 1,
                "Husband, Wife or Partner in a registered marriage, male same-sex couple": 1,
                "Husband, Wife or Partner in a registered marriage, female same-sex couple": 1,
                "Husband, Wife or Partner in de facto marriage, opposite-sex couple": 1,
                "Husband, Wife or Partner in de facto marriage, male same-sex couple": 1,
                "Husband, Wife or Partner in de facto marriage, female same-sex couple": 1,
                "Lone parent": 2,
                "Natural or adopted child under 15": 3,
                "Step child under 15": 3,
                "Foster child under 15": 3,
                "Grandchild under 15": 3,
                "Otherwise related child under 15": 3,
                "Unrelated child under 15": 3,
                "Natural or adopted dependent student": 4,
                "Dependent student step child": 4,
                "Dependent student foster child": 4,
                "Non-dependent natural, or adopted child": 5,
                "Non-dependent step child": 5,
                "Non-dependent foster child": 5,
                "Brother/sister": 6,
                "Father/mother": 6,
                "Non-dependent grandchild": 6,
                "Grandfather/grandmother": 6,
                "Cousin": 6,
                "Uncle/aunt": 6,
                "Nephew/niece": 6,
                "Other related individual (nec)": 6,
                "Unrelated individual living in family household": 7,
                "Group household member": 7,
                "Lone person": 7,
                "Visitor (from within Australia)": 9,
                "Other non-classifiable relationship": 9,
                "Not applicable": 11,
                "Overseas visitor": 11
            },


            # Mapping for person categories
        "nprd": {
            "One person": 1,
            "Two persons": 2,
            "Three persons": 3,
            "Four persons": 4,
            "Five persons": 5,
            "Six persons": 6,
            "Not applicable":7
        },

        # Mapping for income ranges
         "hind" : {
            "Negative income": 1,
            "Nil income": 2,
            "$1-$149 ($1-$7,799)": 3,
            "$150-$299 ($7,800-$15,599)": 4,
            "$300-$399 ($15,600-$20,799)": 5,
            "$400-$499 ($20,800-$25,999)": 6,
            "$500-$649 ($26,000-$33,799)": 7,
            "$650-$799 ($33,800-$41,599)": 8,
            "$800-$999 ($41,600-$51,999)": 9,
            "$1,000-$1,249 ($52,000-$64,999)": 10,
            "$1,250-$1,499 ($65,000-$77,999)": 11,
            "$1,500-$1,749 ($78,000-$90,999)": 12,
            "$1,750-$1,999 ($91,000-$103,999)": 13,
            "$2,000-$2,499 ($104,000-$129,999)": 14,
            "$2,500-$2,999 ($130,000-$155,999)": 15,
            "$3,000-$3,499 ($156,000-$181,999)": 16,
            "$3,500-$3,999 ($182,000-$207,999)": 17,
            "$4,000-$4,499 ($208,000-$233,999)": 18,
            "$4,500-$4,999 ($234,000-$259,999)": 19,
            "$5,000-$5,999 ($260,000-$311,999)": 20,
            "$6,000-$7,999 ($312,000-$415,999)": 21,
            "$8,000 or more ($416,000 or more)": 22,
            "Partial income stated": 23,
            "All incomes not stated": 24,
            "Not applicable": 25
         },

        # Mapping for mortgage repayments
        "mrerd" : {
            "Nil repayments": 1,
            "$1-$149": 2,
            "$150-$299": 3,
            "$300-$449": 4,
            "$450-$599": 5,
            "$600-$799": 6,
            "$800-$999": 7,
            "$1,000-$1,199": 8,
            "$1,200-$1,399": 9,
            "$1,400-$1,599": 10,
            "$1,600-$1,799": 11,
            "$1,800-$1,999": 12,
            "$2,000-$2,199": 13,
            "$2,200-$2,399": 14,
            "$2,400-$2,599": 15,
            "$2,600-$2,799": 16,
            "$2,800-$2,999": 17,
            "$3,000-$3,499": 18,
            "$3,500-$3,999": 19,
            "$4,000-$4,999": 20,
            "$5,000 and over": 21,
            "Not stated": 22,
            "Not applicable": 23
        },

        # mappign for strd
        "strd":{
            "Separate house": 1,
            "Semi-detached, row or terrace house, townhouse etc. with one storey": 2,
            "Semi-detached, row or terrace house, townhouse etc. with two or more storeys": 2,
            "Flat or apartment in a one or two storey block": 3,
            "Flat or apartment in a three storey block": 3,
            "Flat or apartment in a four to eight storey block": 3,
            "Flat or apartment in a nine or more storey block": 3,
            "Flat or apartment attached to a house": 3,
            "Caravan": 9,
            "Cabin, houseboat": 9,
            "Improvised home, tent, sleepers out": 9,
            "House or flat attached to a shop, office, etc.": 9,
            "Not stated": 4,
            "Not applicable": 5
        },
        "qallp":{
            "Postgraduate Degree Level": 1,
            "Graduate Diploma and Graduate Certificate Level": 2,
            "Bachelor Degree Level": 3,
            "Advanced Diploma and Diploma Level": 4,
            "Certificate Level": 5,
            "Level of education inadequately described": 6,
            "Level of education not stated": 7,
            "Not applicable": 8,
            "Overseas visitor": 9
        },
        "qalfp":{
            "Natural and Physical Sciences": 1,
            "Information Technology": 2,
            "Engineering and Related Technologies": 3,
            "Architecture and Building": 4,
            "Agriculture, Environmental and Related Studies": 5,
            "Health": 6,
            "Education": 7,
            "Management and Commerce": 8,
            "Society and Culture": 9,
            "Creative Arts": 10,
            "Food, Hospitality and Personal Services": 11,
            "Mixed Field Programmes": 12,
            "Field of study inadequately described": 13,
            "Field of study not stated": 14,
            "Not applicable": 15,
            "Overseas visitor": 16
        },
        "incp":{
                "Negative income": 1,
                "Nil income": 2,
                "$1-$149 ($1-$7,799)": 3,
                "$150-$299 ($7,800-$15,599)": 4,
                "$300-$399 ($15,600-$20,799)": 5,
                "$400-$499 ($20,800-$25,999)": 6,
                "$500-$649 ($26,000-$33,799)": 7,
                "$650-$799 ($33,800-$41,599)": 8,
                "$800-$999 ($41,600-$51,999)": 9,
                "$1,000-$1,249 ($52,000-$64,999)": 10,
                "$1,250-$1,499 ($65,000-$77,999)": 11,
                "$1,500-$1,749 ($78,000-$90,999)": 12,
                "$1,750-$1,999 ($91,000-$103,999)": 13,
                "$2,000-$2,999 ($104,000-$155,999)": 14,
                "$3,000-$3,499 ($156,000-$181,999)": 15,
                "$3,500 or more ($182,000 or more)": 16,
                "Not stated": 17,
                "Not applicable": 18,
                "Overseas visitor": 19
            },
            "hrsp": {
                "0 hours": 1,
                "1 hour": 2,
                "2 hours": 3,
                "3 hours": 4,
                "4 hours": 5,
                "5 hours": 6,
                "6 hours": 7,
                "7 hours": 8,
                "8 hours": 9,
                "9 hours": 10,
                "10 hours": 11,
                "11 hours": 12,
                "12 hours": 13,
                "13 hours": 14,
                "14 hours": 15,
                "15 hours": 16,
                "16 hours": 17,
                "17 hours": 18,
                "18 hours": 19,
                "19 hours": 20,
                "20 hours": 21,
                "21 hours": 22,
                "22 hours": 23,
                "23 hours": 24,
                "24 hours": 25,
                "25 hours": 26,
                "26 hours": 27,
                "27 hours": 28,
                "28 hours": 29,
                "29 hours": 30,
                "30 hours": 31,
                "31 hours": 32,
                "32 hours": 33,
                "33 hours": 34,
                "34 hours": 35,
                "35 hours": 36,
                "36 hours": 37,
                "37 hours": 38,
                "38 hours": 39,
                "39 hours": 40,
                "40 hours": 41,
                "41 hours": 42,
                "42 hours": 43,
                "43 hours": 44,
                "44 hours": 45,
                "45 hours": 46,
                "46 hours": 47,
                "47 hours": 48,
                "48 hours": 49,
                "49 hours": 50,
                "50 hours": 51,
                "51 hours": 52,
                "52 hours": 53,
                "53 hours": 54,
                "54 hours": 55,
                "55 hours": 56,
                "56 hours": 57,
                "57 hours": 58,
                "58 hours": 59,
                "59 hours": 60,
                "60 hours": 61,
                "61 hours": 62,
                "62 hours": 63,
                "63 hours": 64,
                "64 hours": 65,
                "65 hours": 66,
                "66 hours": 67,
                "67 hours": 68,
                "68 hours": 69,
                "69 hours": 70,
                "70 hours": 71,
                "71 hours": 72,
                "72 hours": 73,
                "73 hours": 74,
                "74 hours": 75,
                "75 hours": 76,
                "76 hours": 77,
                "77 hours": 78,
                "78 hours": 79,
                "79 hours": 80,
                "80 hours": 81,
                "81 hours": 82,
                "82 hours": 83,
                "83 hours": 84,
                "84 hours": 85,
                "85 hours": 86,
                "86 hours": 87,
                "87 hours": 88,
                "88 hours": 89,
                "89 hours": 90,
                "90 hours": 91,
                "91 hours": 92,
                "92 hours": 93,
                "93 hours": 94,
                "94 hours": 95,
                "95 hours": 96,
                "96 hours": 97,
                "97 hours": 98,
                "98 hours": 99,
                "99 hours": 100,
                "less than 1 hour na": 1,
                "1 - 15 h": 2,
                "16 - 24": 3,
                "25 - 34": 4,
                "35 - 39": 5,
                "40 hours": 6,
                "41 - 49": 7,
                "50 and over": 8,
                "Not stated": 101,
                "Not applicable": 102,
                "Overseas visitor": 103
            },
            "hrsp_m":{
            "Less than 1 hour and Not applicable":1,
            "1-15 hours":2,
            "16-24 hours":3,
            "25-34 hours":4,
            "35-39 hours":5,
            "40 hours":6,
            "41-49 hours":7,
            "50 hours and over":8,
            "Not stated":9,
            "Overseas visitor":10,
            # Abbreviated labels used by sa2hrsp.csv statewide ABS files
            "less than 1 hour na": 1,
            "1 - 15 h": 2,
            "16 - 24": 3,
            "25 - 34": 4,
            "35 - 39": 5,
            "41 - 49": 7,
            "50 and over": 8,
            # Numeric codes produced when the individual-hour hrsp mapping runs
            # before hrsp_m (e.g. "40 hours"->41, "Not stated"->101, "Overseas visitor"->103)
            41: 6,
            101: 9,
            102: 1,
            103: 10,
            }


        ,
            "grgp":
                {
                    "National Government": 1,
                    "State/Territory Government": 2,
                    "Local Government": 3,
                    "Private sector": 4,
                    "Not stated": 5,
                    "Not applicable": 6,
                    "Overseas visitor": 7
                },
        "empp":{
            "Nil employees": 1,
        "1-19 employees": 2,
        "20 or more employees": 3,
        "Not stated": 4,
        "Not applicable": 5,
        "Overseas visitor": 6
        },
        "tenlld":{
            "Owned outright": 1,
            "Owned with a mortgage": 2,
            "Rented: Real estate agent": 3,
            "Rented: State or territory housing authority": 4,
            "Rented: Community housing provider": 5,
            "Rented: Person not in same household": 6,
            "Rented: Other landlord type": 7,
            "Rented: Landlord type not stated": 8,
            "Other tenure type": 9,
            "Tenure type not stated": 10,
            "Tenure type not applicable": 11
        },
        "stup":{
            "Not attending": 1,
            "Full-time student": 2,
            "Part-time student": 3,
            "Institution (TYPP) stated, full-time/part-time status (STUP) not stated": 4,
            "Both not stated - both institution (TYPP) and full-time/part-time status (STUP) not stated": 5,
            "Overseas visitor": 6,
            # Abbreviated label used in sa2stup.csv statewide ABS file
            "Not Stated G": 5,
        },
            "siemp": {
                "Employee": 1,
                "Owner manager of incorporated enterprise with employees": 2,
                "Owner manager of incorporated enterprise without employees": 3,
                "Owner manager of incorporated enterprise - employees not stated": 4,
                "Owner manager of unincorporated enterprise with employees": 5,
                "Owner manager of unincorporated enterprise without employees": 6,
                "Owner manager of unincorporated enterprise - employees not stated": 7,
                "Contributing family worker": 8,
                "Not stated": 9,
                "Not applicable": 10,
                "Overseas visitor": 11
            },


        # Mapping for rent ranges
        "rntrd" : {
            "$1-$74": 1,
            "$75-$99": 2,
            "$100-$124": 3,
            "$125-$149": 4,
            "$150-$174": 5,
            "$175-$199": 6,
            "$200-$224": 7,
            "$225-$249": 8,
            "$250-$274": 9,
            "$275-$299": 10,
            "$300-$324": 11,
            "$325-$349": 12,
            "$350-$374": 13,
            "$375-$399": 14,
            "$400-$424": 15,
            "$425-$449": 16,
            "$450-$474": 17,
            "$475-$499": 18,
            "$500-$524": 19,
            "$525-$549": 20,
            "$550-$649": 21,
            "$650-$749": 22,
            "$750-$849": 23,
            "$850-$949": 24,
            "$950 and over": 25,
            "Not stated": 26,
            "Not applicable": 27
        },




        # Mapping for education levels
        "heap": {
            "Postgraduate Degree Level": 1,
            "Graduate Diploma and Graduate Certificate Level": 2,
            "Bachelor Degree Level": 3,
            "Advanced Diploma and Diploma Level": 4,
            "Certificate III & IV Level": 5,
            "Secondary Education - Years 10 and above": 6,
            "Certificate I & II Level": 7,
            "Secondary Education - Years 9 and below": 8,
            "Supplementary Codes": 9,
            "Not stated": 10,
            "Not applicable": 11,
            "Overseas visitor": 12
        },

        # Mapping for travel modes
        "mtwp" : {
            "Train": 1,
            "Bus": 2,
            "Ferry": 3,
            "Tram/light rail": 4,
            "Taxi/ride-share service": 5,
            "Car, as driver": 6,
            "Car, as passenger": 7,
            "Truck": 8,
            "Motorbike/scooter": 9,
            "Bicycle": 10,
            "Walked only": 11,
            "Other Mode": 12,
            "Worked at home": 13,
            "Did not go to work": 14,
            "Not stated": 15,
            "Not applicable": 16,
            "Overseas visitor": 17},


        # SEXP
        "sexp":{
                "Male":1,
                "Female":2
        },

        "mstp":{
              "Never married": 1,
              "Widowed" : 2,
              "Divorced" : 3,
              "Separated" : 4,
              "Married" : 5,
              "Not applicable" :6
            }
        }
        self.reverse_mappings = {}
        for map_name, mapping in self.mappings.items():
            rev = {}
            for k, v in mapping.items():
                if isinstance(v, list):
                    for item in v:
                        rev[item] = k
                else:
                    rev[v] = k
            self.reverse_mappings[map_name] = rev


    def _apply_mapping(self, column_name, mapping):
        """
        Applies a mapping to a specific column in the control table.

        Parameters:
            column_name (str): The name of the column to map.
            mapping (dict): The mapping dictionary.

        Returns:
            None
        """
        if column_name not in self.control_table.columns:
            logging.debug(f"Column '{column_name}' not found in the control table. Skipping mapping.")
            return

        # Apply the mapping and replace the column values.
        # Some mapping dicts have list values (e.g. age5p: "0-4 years" → [0,1,2,3,4])
        # which are intended for microdata expansion.  For control-table aggregation we
        # need a single scalar code per label — use the minimum of any list value.
        scalar_mapping = {
            k: (min(v) if isinstance(v, list) else v)
            for k, v in mapping.items()
        }
        self.control_table[column_name] = self.control_table[column_name].map(scalar_mapping).fillna(
            self.control_table[column_name])
        other_key_cols = [c for c in self.control_table.columns
                          if c not in [self.sa_num, column_name, 'count']
                          and not c.startswith('unnamed')]
        group_cols = [self.sa_num] + other_key_cols + [column_name]
        self.control_table = (
            self.control_table
            .groupby(group_cols, as_index=False)
            .agg({'count': 'sum'})
        )
    def _read_and_process_table(self):
        """
        Reads and processes the control table from the CSV file.

        Returns:
            pd.DataFrame: The processed control table.
        """
        # Check if the file exists and is readable
        if not os.path.isfile(self.file_path) or not os.access(self.file_path, os.R_OK):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist or cannot be read.")

        # Determine start and end rows for the data
        start_and_end_row_indexes = self._find_start_and_end_row_indexes()
        logging.debug(f"Start and end indexes: {start_and_end_row_indexes}")

        # Read the CSV file with dynamic row selection
        try:
            control_table = pd.read_csv(
                self.file_path,
                skiprows=start_and_end_row_indexes[0],
                nrows=start_and_end_row_indexes[1] - start_and_end_row_indexes[0],
                engine="python",
            )
        except Exception as e:
            logging.warning(f"Retrying {self.file_path} after parse error: {e}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                control_table = pd.read_csv(
                    self.file_path,
                    skiprows=start_and_end_row_indexes[0],
                    nrows=start_and_end_row_indexes[1] - start_and_end_row_indexes[0],
                    engine="python",
                )
                # Drop the last column if a mismatch occurs
                control_table = control_table.iloc[:, :-1]

        # Remove rows with 'Total' if specified
        if self.remove_totals:
            control_table = control_table[~control_table.isin(["Total"]).any(axis=1)]

        # Simplify column names if specified
        if self.simplify_names:
            control_table.columns = (
                control_table.columns
                .str.replace(r"\s.*", "", regex=True)  # Remove everything after the first space
                .str.strip()                          # Remove leading/trailing whitespace
                .str.replace(" ", "_")                # Replace spaces with underscores
                .str.lower()                          # Convert to lowercase
            )
        return control_table

    def _find_start_and_end_row_indexes(self):
        """
        Determines the start and end row indexes for the data in the CSV file.

        Returns:
            tuple: Start and end row indexes as (start_row_index, end_row_index).
        """
        with open(self.file_path, "r") as file:
            count_fields = [len(line.strip().split(",")) for line in file]

        # Find the first row where the number of fields is greater than 2
        start_row_index = next((i for i, count in enumerate(count_fields) if count > 2), None)
        if start_row_index == 0:
            # Clean header-first file (e.g. generated joint controls):
            # the header really is row 0.  ABS TableBuilder extracts carry
            # an 8-line preamble, in which case start_row_index is 8, not 0.
            start_row_index = 0
        elif start_row_index is None:
            start_row_index = 9
            raise ValueError("The start row index cannot be determined.")

        # Find the first row after start_row_index where the number of fields is 1
        end_row_index = next(
            (i for i, count in enumerate(count_fields[start_row_index + 1:], start=start_row_index + 1) if count == 1),
            len(count_fields),
        ) - 1

        if end_row_index < start_row_index:
            raise ValueError("The end row index cannot be determined.")

        return (start_row_index, end_row_index)

    def condense_from_mapping(self, mapa, mapb, output):
        """
        Condenses the data from `mapa` to `output` using the mapping provided in `mapb`.




        Parameters:
        - mapa: A dictionary where keys are detailed relationship descriptions and values are counts or IDs.
        - mapb: A mapping dictionary that maps condensed categories to lists of IDs from `mapa`.
        - output: A dictionary where the condensed results will be stored.

        Returns:
        - output: A dictionary containing counts or values for the condensed categories.
        """
        # Initialize the output dictionary with all keys from `mapb` and set their values to 0

        self.control_table
        #conends control_table


        for condensed_category in self.mappings[mapb]:
            output[condensed_category] = 0

        # Iterate through the mapping in `mapb` and condense the values from `mapa`
        for condensed_category, ids in self.mappings[mapb].items():
            for id_ in ids:
                if id_ in self.mappings[mapa]:  # Check if the ID exists in `mapa`
                    output[condensed_category] += mapa[id_]

        return output

    def validate_and_clean(self):
        """
        Validates and cleans the control table.

        Returns:
            pd.DataFrame: The validated and cleaned control table.
        """

        for column, mapping in self.mappings.items():
            # Only attempt the mapping if the column is actually present in this table.
            # Each CSV covers a single attribute, so most mappings are irrelevant — skip
            # them silently rather than logging or warning for every mismatched pair.
            if column in self.control_table.columns:
                self._apply_mapping(column, mapping)

        if not isinstance(self.control_table, pd.DataFrame):
            raise ValueError("control_table must be a pandas DataFrame.")
        if len(self.control_table) < 1 or self.control_table.shape[1] < 2:
            raise ValueError("control_table must have at least 1 row and 2 columns.")
        if "count" not in self.control_table.columns:
            raise ValueError("control_table must include a 'count' column.")
        if not pd.api.types.is_integer_dtype(self.control_table["count"]):
            raise ValueError("'count' column must contain integer-like values.")
        if (self.control_table["count"] < 0).any():
            raise ValueError("'count' column must not contain negative values.")
        if self.control_table["count"].isnull().any():
            raise ValueError("'count' column must not contain missing values.")

        # If "counting" column exists, remove it
        if "counting" in self.control_table.columns:
            self.control_table = self.control_table.drop(columns=["counting"])

        if 'unnamed:' in self.control_table.columns:
            self.control_table = self.control_table.drop(columns=["unnamed:"])
            # remove the unnamed: column, if all nans, otherwse raide exceptions

        if 'age5p' in self.control_table:
            grouped = (
                self.control_table
                .groupby([self.sa_num, 'age5p'], as_index=False)
                .agg({'count': 'sum'})  # Replace 'count' with the actual column for counts
            )
            # Update the control_table with the grouped results
            self.control_table = grouped

        if 'rlhp' in self.control_table:
            grouped = (
                self.control_table
                .groupby([self.sa_num, 'rlhp'], as_index=False)
                .agg({'count': 'sum'})  # Replace 'count' with the actual column for counts
            )
            # Update the control_table with the grouped results
            self.control_table = grouped

        if 'strd' in self.control_table:
            grouped = (
                self.control_table
                .groupby([self.sa_num, 'strd'], as_index=False)
                .agg({'count': 'sum'})  # Replace 'count' with the actual column for counts
            )
            self.control_table = grouped

        return self.control_table

    def get_table(self):
        """
        Returns the processed and validated control table.

        Returns:
            pd.DataFrame: The control table.
        """
        return self.control_table

    def __repr__(self):
        """
        String representation of the CensusControlTable object.
        """
        return f"CensusControlTable(file_path='{self.file_path}', rows={len(self.control_table)})"



class HouseholdTS(CensusControlTable):
    def __init__(self):
        super().__init__()


    def define_mapping(self):
        # re
        self.mappings ={
            "income": {
                "Select One":18,
                "Prefer not so say / can't say":17,
                "Nil or negative income":[1,2],
                "$1 - $149 per week ($1 - $7,799 per year)":3,
                "$150 - $299 per week ($7,800 - $15,599 per year)":4,
                "$300 - $399 per week ($15,600 - $20,799 per year)":5,
                "$400 - $499 per week ($20,800 - $25,999 per year)":6,
                "$500 - $649 per week ($26,000 - $33,799 per year)":7,
                "$650 - $799 per week ($33,800 - $41,599 per year)":8,
                "$800 - $999 per week ($41,600 - $51,999 per year)":9,
                "$1,250 - $1,499 per week ($65,000 - $77,999 per year)":11,
                "$1,000 - $1,249 per week ($52,000 - $64,999 per year)":10,
                "$1,500 - $1,749 per week ($78,000 - $90,999 per year)":12,
                "$1,750 - $1,999 per week ($91,000 - $103,999 per year)":13,
                "$2,000 - $2,999 per week ($104,000 - $155,999 per year)":14,
                "$3,000 - $3,499 per week ($156,000 - $181,999 per year)":15,
                "$3,500 or more per week ($182,000 or more per year)":16

            },
            "SEX" :{"male":1,
                    "female":2},
           "STUDYING":{"typp": {
                "Preschool": 1,
                "Primary - Government": 2,
                "Primary - Catholic": 3,
                "Primary - other non-Government": 3,
                "Primary - not further defined": 3,
                "Secondary - Government": 4,
                "Secondary - Catholic": 5,
                "Secondary - other non-Government": 5,
                "Secondary - not further defined": 5,
                "Tertiary - Vocational education (including TAFE and private training providers)": 6,
                "Tertiary - University or other higher education": 7,
                "Tertiary - not further defined": 6,
                "Other": 8,
                "Not stated": 9,
                "Not applicable": 10,
                "Overseas visitor": 11
            },"tertiaryPartTime": [6,7,8],
               "tertiaryFullTime": [6,7,8],
               "secondary":[4,5],
               "primary":[2,3],
                "childcare":1,
                "other": 8},

            "WORKSTATUS":{
                "workFullTime":1,
                "otherReason":1,
                "workPartTime":1,
                "carer":1,
                "retired":1,
                "longTermIllness":1,
                "shortTermIllness":1
            },

        "industry": { "accom": 8,
                      "publicAdmin": 15,
                      "construction": 2,
                      "health": 17,
                      "it": 10,
                      "transport": 9,
                      "construction":5,
                      "education": 16,
                        "manufacturing":3,
                        "financial": 11,
                      "other": 19,
                      "arts": 18,
                      "wholesale": 6,
                      "retail": 7,
                      "utilities": 4,
                      "realEstate": 12,
                      "mining": 2,
                      "professional": 13,
                      "admin": 14,
                      "agriculture":1






                      } #occp
            ,
            "DWELLTYPE":{"Separate house":1,
            "Unit, flat, or apartment (less than 4 storeys)":3,
            "Semi-detached house, terrace house, townhouse, etc.":2,
            "Unit, flat, or apartment(4 or more storeys)":3
                         },


            "RELATIONSHIP":{
                "spouse":1,
                "child":1,
                "nonFamily":1,
                "relative":1
        }



        }


class SA_1_CensusControlTable(CensusControlTable):
    def __init__(self, file_path,  remove_totals=True, simplify_names=True, *args, **kwargs):
        super().__init__(file_path,  remove_totals=remove_totals, simplify_names=simplify_names, *args, **kwargs)
        self.sa_num = kwargs.get('zone_level')
        self.validate_and_clean()
        
        
        
    
    
    
    def define_mapping(self):
        # Define mapping
        self.mappings = {
            
            #CORE ACTIVITY FOR ASSISTANCE
        "assnp":{"Has need for assistance with core activities":1, 
                 "Does not have need for assistance with core activities":2,
                 "Not stated":3, 
                 "Overseas visitor":4},
        'agep': {
            '0-4 years': 1,
            '5-9 years':2,
            '10-14 years':3,
            '15-19 years':4,
            '20-24 years':5,
            '25-29 years':6,
            '30-34 years':7,
            '35-39 years':8,
            '40-44 years':9,
            '45-49 years':10,
            '50-54 years':11,
            '55-59 years':12,
            '60-64 years':13,
            '65-69 years':14,
            '70-74 years':15,
            '75-79 years':16,
            '80-84 years':17,
            '85 years and over':18,
            '90-94 years':19,
            '95-99 years':20,
            '100 years and over':21        
        },
        'age5p': {
            '0-4 years': 1,
            '5-9 years':2,
            '10-14 years':3,
            '15-19 years':4,
            '20-24 years':5,
            '25-29 years':6,
            '30-34 years':7,
            '35-39 years':8,
            '40-44 years':9,
            '45-49 years':10,
            '50-54 years':11,
            '55-59 years':12,
            '60-64 years':13,
            '65-69 years':14,
            '70-74 years':15,
            '75-79 years':16,
            '80-84 years':17,
            '85-89 years':18,
            '90-94 years':18,
            '95-99 years':18,
            '100 years and over':18        
        },
        
        'bedrd':{
            'None (includes studio apartments or bedsitters)':0,
            'One bedroom':1,
            'Two bedrooms':2,
            'Three bedrooms':4,
            'Four bedrooms':5,
            'Five bedrooms or more':6,
            'Not stated':6,
            'Not applicable':7
        },
        'number_of_families':{
         'One family household':1,
        'Two family household':2,
        'Three or more family household':3
        },
        'heap':{
            'Postgraduate Degree Level':1,
            'Graduate Diploma and Graduate Certificate Level':2,
            'Bachelor Degree Level':3,
            'Advanced Diploma and Diploma Level':4,
            'Certificate III & IV Level':5,
            'Secondary Education - Years 10 and above':6,
            'Certificate I & II Level':7,
            'Secondary Education - Years 9 and below':8,
            'Supplementary Codes':9,
            'Not stated':10,
            'Not applicable':11,
            'Overseas visitor':12
        },
        'hhcd':{'One family household':1,
                'Two or more family household':2,
                'Lone person household':3,
                'Group household':4,
                'Visitors only':5,
                'Other non-classifiable':6,
                'Not applicable':7
            
            },
        'hind':{'Negative income':1,
                'Nil income':2,
                '$1-$149 ($1-$7,799)':3,
                '$150-$299 ($7,800-$15,599)':4,
                '$300-$399 ($15,600-$20,799)':5,
                '$400-$499 ($20,800-$25,999)':6,
                '$500-$649 ($26,000-$33,799)':7,
                '$650-$799 ($33,800-$41,599)':8,
                '$800-$999 ($41,600-$51,999)':9,
                '$1,000-$1,249 ($52,000-$64,999)':10,
                '$1,250-$1,499 ($65,000-$77,999)':11,
                '$1,500-$1,749 ($78,000-$90,999)':12,
                '$1,750-$1,999 ($91,000-$103,999)':13,
                '$2,000-$2,499 ($104,000-$129,999)':14,
                '$2,500-$2,999 ($130,000-$155,999)':15,
                '$3,000-$3,499 ($156,000-$181,999)':16,
                '$3,500-$3,999 ($182,000-$207,999)':17,
                '$4,000-$4,499 ($208,000-$233,999)':18,
                '$4,500-$4,999 ($234,000-$259,999)':19,
                '$5,000-$5,999 ($260,000-$311,999)':20,
                '$6,000-$7,999 ($312,000-$415,999)':21,
                '$8,000 or more ($416,000 or more)':22, 
                'Partial income stated':23,
                'All incomes not stated':24,   
                'Not applicable':25
        },
        'hrsp':{
            'less than 1 hour na':1,
            '1 - 15 h':2,
            '16 - 24':3,
            '25 - 34':4,
            '35 - 39':5,
            '41 - 49':7,
            '50 and over':8,
            '40 hours':6,
            'Not stated':9,
            'Overseas visitor':10
        },
        "incp":{
                "Negative income": 1,
                "Nil income": 2,
                "$1-$149 ($1-$7,799)": 3,
                "$150-$299 ($7,800-$15,599)": 4,
                "$300-$399 ($15,600-$20,799)": 5,
                "$400-$499 ($20,800-$25,999)": 6,
                "$500-$649 ($26,000-$33,799)": 7,
                "$650-$799 ($33,800-$41,599)": 8,
                "$800-$999 ($41,600-$51,999)": 9,
                "$1,000-$1,249 ($52,000-$64,999)": 10,
                "$1,250-$1,499 ($65,000-$77,999)": 11,
                "$1,500-$1,749 ($78,000-$90,999)": 12,
                "$1,750-$1,999 ($91,000-$103,999)": 13,
                "$2,000-$2,999 ($104,000-$155,999)": 14,
                "$3,000-$3,499 ($156,000-$181,999)": 15,
                "$3,500 or more ($182,000 or more)": 16,
                "Not stated": 17,
                "Not applicable": 18,
                "Overseas visitor": 19
            },
        "indp": {
                "Agriculture, Forestry and Fishing": 1,
                "Mining": 2,
                "Manufacturing": 3,
                "Electricity, Gas, Water and Waste Services": 4,
                "Construction": 5,
                "Wholesale Trade": 6,
                "Retail Trade": 7,
                "Accommodation and Food Services": 8,
                "Transport, Postal and Warehousing": 9,
                "Information Media and Telecommunications": 10,
                "Financial and Insurance Services": 11,
                "Rental, Hiring and Real Estate Services": 12,
                "Professional, Scientific and Technical Services": 13,
                "Administrative and Support Services": 14,
                "Public Administration and Safety": 15,
                "Education and Training": 16,
                "Health Care and Social Assistance": 17,
                "Arts and Recreation Services": 18,
                "Other Services": 19,
                "Inadequately described": 20,
                "Not stated": 21,
                "Not applicable": 22,
                "Overseas visitor": 23
            },
        'lfsp':{
            'Employed':1,
            'Unemployed':2,
            'Not in the labour force':3,
            'Not stated':4,
            'Not applicable':5,
            'Overseas visitor':6,
        },
        
        "mrerd" : {
            "Nil repayments": 1,
            "$1-$149": 2,
            "$150-$299": 3,
            "$300-$449": 4,
            "$450-$599": 5,
            "$600-$799": 6,
            "$800-$999": 7,
            "$1,000-$1,199": 8,
            "$1,200-$1,399": 9,
            "$1,400-$1,599": 10,
            "$1,600-$1,799": 11,
            "$1,800-$1,999": 12,
            "$2,000-$2,199": 13,
            "$2,200-$2,399": 14,
            "$2,400-$2,599": 15,
            "$2,600-$2,799": 16,
            "$2,800-$2,999": 17,
            "$3,000-$3,499": 18,
            "$3,500-$3,999": 19,
            "$4,000-$4,999": 20,
            "$5,000 and over": 21,
            "Not stated": 22,
            "Not applicable": 23
        },
       

        # Mapping for travel modes
        "mtwp" : {
            "Train": 1,
            "Bus": 2,
            "Ferry": 3,
            "Tram/light rail": 4,
            "Taxi/ride-share service": 5,
            "Car, as driver": 6,
            "Car, as passenger": 7,
            "Truck": 8,
            "Motorbike/scooter": 9,
            "Bicycle": 10,
            "Walked only": 11,
            "Other Mode": 12,
            "Worked at home": 13,
            "Did not go to work": 14,
            "Not stated": 15,
            "Not applicable": 16,
            "Overseas visitor": 17},


        # SEXP
        "sexp":{
                "Male":1,
                "Female":2
        },

        "mstp":{
              "Never married": 1,
              "Widowed" : 2,
              "Divorced" : 3,
              "Separated" : 4,
              "Married" : 5,
              "Not applicable" :6
        },
         "nprd": {
            "One person": 1,
            "Two persons": 2,
            "Three persons": 3,
            "Four persons": 4,
            "Five persons": 5,
            "Six persons": 6,           # ABS 2021 Census label (was incorrectly "Six persons or more")
            "Seven persons": 7,
            "Eight or more persons": 8,
            "Eight persons or more": 8,
            "Not stated": None,         # Exclude from SA targets (imputed separately)
            "Not applicable": None,     # Non-private dwellings: exclude from control targets
            "Overseas visitor": None,
        },

         "vehrd": {
            "No motor vehicles": 0,
            "One motor vehicle": 1,
            "Two motor vehicles": 2,
            "Three motor vehicles": 3,
            "Four or more motor vehicles": 4,
            "Not stated": None,         # Exclude: no code in microdata → SA would never satisfy
            "Not applicable": None,     # Exclude: non-private dwellings
        },

        "occp": {
            "Managers": 1, "Professionals": 2, "Technicians and Trades Workers": 3,
            "Community and Personal Service Workers": 4,
            "Clerical and Administrative Workers": 5, "Sales Workers": 6,
            "Machinery Operators and Drivers": 7, "Labourers": 8,
            "Inadequately described": 9, "Not stated": 10,
            "Not applicable": 11,
        },
        "qallp":{
            "Postgraduate Degree Level": 1,
            "Graduate Diploma and Graduate Certificate Level": 2,
            "Bachelor Degree Level": 3,
            "Advanced Diploma and Diploma Level": 4,
            "Certificate Level": 5,
            "Level of education inadequately described": 6,
            "Level of education not stated": 7,
            "Not applicable": 8,
            "Overseas visitor": 9
        },
        "qalfp":{
            "Natural and Physical Sciences": 1,
            "Information Technology": 2,
            "Engineering and Related Technologies": 3,
            "Architecture and Building": 4,
            "Agriculture, Environmental and Related Studies": 5,
            "Health": 6,
            "Education": 7,
            "Management and Commerce": 8,
            "Society and Culture": 9,
            "Creative Arts": 10,
            "Food, Hospitality and Personal Services": 11,
            "Mixed Field Programmes": 12,
            "Field of study inadequately described": 13,
            "Field of study not stated": 14,
            "Not applicable": 15,
            "Overseas visitor": 16
        },
        'rlhp':{
            'Husband Wife or Partner':1,
            'Child und 15':3,
            'Dep student':4,
            'Non-dep student':5,
            'Other related':6,
            'Non family member':7,
            'Non classifiable member':9,
            'Lone parent':2,
            'Not applicable':10,
            'Overseas visitor':11,

        },
         "stup":{
            "Not attending": 1,
            "Full-time student": 2,
            "Part-time student": 3,
            "Not Stated G": 4,
            "Overseas visitor": 5
        },
        "tend":{
            'Owned with a mortgage':2,
            'Rented':3,
            'Other':4,
            'Not stated':5,
            'Owned outright':1,
            'Not applicable':6,

        },
        'typp':{
            'Primary - Non Government':3,
            'Secondary - Non-Government':5,
            'Tertiary - Non University':6,
            'Preschool':1,
            'Primary - Government':2,
            'Secondary - Government':4,
            'Tertiary - University or other higher education':7,
            'Other':8,
            'Not stated':9,
            'Not applicable':10,
            'Overseas visitor':11
        },
        
        
        "rntrd" : {
            "$1-$74": 1,
            "$75-$99": 2,
            "$100-$124": 3,
            "$125-$149": 4,
            "$150-$174": 5,
            "$175-$199": 6,
            "$200-$224": 7,
            "$225-$249": 8,
            "$250-$274": 9,
            "$275-$299": 10,
            "$300-$324": 11,
            "$325-$349": 12,
            "$350-$374": 13,
            "$375-$399": 14,
            "$400-$424": 15,
            "$425-$449": 16,
            "$450-$474": 17,
            "$475-$499": 18,
            "$500-$524": 19,
            "$525-$549": 20,
            "$550-$649": 21,
            "$650-$749": 22,
            "$750-$849": 23,
            "$850-$949": 24,
            "$950 and over": 25,
            "Not stated": 26,
            "Not applicable": 27
        },
        
        'strd': {'Semi-detached':2,
                    'Flat or apartment':3,
                    'Other dwelling':4,
                    'Not stated':5,
                    'Not applicable':6,
                    'Separate house':1,

            
        }
                 
        
        
        }
        






# Example Usage
if __name__ == "__main__":
    # Path to the control table CSV file
    csv_file_path = "data_sa1_controls/sa1lfsp.csv"
  

    # Initialize the CensusControlTable object
    control_table = SA_1_CensusControlTable(csv_file_path)
    print(control_table)
    # Validate and clean the table
    validated_table = control_table.validate_and_clean()

    # Display the processed control table
    print("Processed Control Table:")
    print(validated_table)
    print('1')