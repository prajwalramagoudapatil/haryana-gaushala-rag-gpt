import pdfplumber
import pandas as pd
from langchain_core.documents import Document
import re

PDF_PATH = "Haryana-Districtwise-Gaushalas Data.pdf"
# CHROMA_PERSIST_DIR = "chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# add_documents = not os.path.exists(CHROMA_PERSIST_DIR)

columns = [
    "Sr. No.",
    "Distt. Sr. No.",
    "Goshala Name",
    "Village",
    "Registration No",
    "Cattle count",
    "Mobile No",
    'Distt Name'
]

def normalize(text):
    text = text.lower()
    text = text.replace("gauraksha", " ")
    text = text.replace("gaushala", " ")
    text = text.replace("goshala", " ")
    terms_to_remove = [
        r'\b(gau|gou|goshala|raksha)\b',
        r'\b(shala|goushala|nandishala)\b', # Added nandishala
        r'\b(trust)\b',
        r'\b(sewa|seva|sewak|sewa\s*samiti|sewa\s*kendra)\b', # Added variations
        r'\b(nandidham|nandi)\b',
        r'\b(samiti|society|welfare|parmarth|cheritable)\b', # Added more institutional terms
        r'\b(avm|evm|v|and|em)\b', # Conjunctions (and)
        # r'\b(shree|shri|krishan|radha|bhagwan|baba|mata|sardar)\b' # Common honorifics/deities (optional removal)
    ]

    normalized_name = text

    # Perform all replacements
    for pattern in terms_to_remove:
        # Use re.sub for case-insensitive replacement (re.IGNORECASE flag)
        normalized_name = re.sub(pattern, ' ', normalized_name, flags=re.IGNORECASE)

    normalized_name = ' '.join(normalized_name.split())

    return normalized_name.strip()



def extract_docs_from_pdf(pdf_path):
    """Extracts tables page-by-page and returns them as text."""
    extracted_docs = []
    distt = ''
    df = pd.DataFrame( columns=columns)
    distt_total = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            if not tables:
                continue
            print(f" Extracting tables from page {page_num}, found {len(tables)} tables.")
            
            for table in tables:
                
                for row in table:
                    
                    cleaned = [r.lower() for r in row if r]
                    
                    if len(cleaned) == (len(columns) - 1):
                        row.append(distt)
                        cleaned_row = [re.sub('\n', ' ', r) for r in row]
                        df.loc[len(df)] = cleaned_row
                    elif any(word in s for s in cleaned for word in ["distt", "district"]):
                        if len(cleaned) <= 3:
                            distt = ' '.join([s for s in row if s is not None])
                        # print(" Row cleaned: ", cleaned)
                        # print(" distt fud ", distt)

                    if any( "total" in r for r in cleaned):
                        txt = ' '.join(cleaned)
                        cattle_count = txt.split()[-1]
                        district_name = ' '.join(str(distt).split()[1:])
                        txt = f" District {district_name} total cattles in {district_name}  {cattle_count}"
                        distt_total.append(txt )
                        print(" Total row found: ", txt)


    print(" Dataframe constructed from PDF: ", df.shape)
    
    for _,row in df.iterrows():
        goshala_words = []
        if row['Goshala Name'] is not None:
            goshala_words = row['Goshala Name'].split()
        owner_name = ''.join([r for r in row['Mobile No'] if not r.isdigit()])
        line = (
            f"Sr No {row['Sr. No.']} Name {row['Goshala Name']} is in {row['Village']} "
            f"has a cattle(cow/ox) count of {row['Cattle count']} has Registration No {row['Registration No']} "
            f"{' '.join(goshala_words)} "
            f"contact deteils {row['Mobile No']}, is in district {row['Distt Name']}, owner {owner_name} {owner_name} \n "
            
        )
        line_doc = Document(
            page_content=(line ),
            metadata={
                "page_label": page_num + 1,
                "type": "text_line",
                "source": pdf_path
            }
        )
        extracted_docs.append(line_doc)
    print(" constructed docs from rows: ", len(extracted_docs))
    for total in distt_total:
        line_doc = Document(
            page_content=(total + '\n' + total + '\n\n'),
            metadata={
                "page_label": page_num + 1,
                "type": "distt_total",
                "source": pdf_path
            }
        )
        extracted_docs.append(line_doc)
    print(" constructed docs from distt totals: ", len(distt_total))
    print("writing extracted docs to text file...")
    with open("extracted_gau_shala_data.txt", "w", encoding="utf-8") as f:
        for doc in extracted_docs:
            f.write(doc.page_content + "\n")

    return

extract_docs_from_pdf(PDF_PATH)


def d():
    if True:
        if False:
            if True:
                df.columns = df.columns.map(lambda x: str(x) if x is not None else "")
                
                # Clean up empty rows/cols usually found in PDFs
                df = df.dropna(how='all')
                
                df = df.loc[:, ~df.columns.str.contains(r"^Unnamed|^None|^nan|^\s*$", case=False)]

                for _, row in df.iterrows():
                    goshala_words = []
                    if row['Goshala Name'] is not None:
                        goshala_words = row['Goshala Name'].split()
                    line = (
                        f"Sr. No. {row['Sr. No.']} Name {row['Goshala Name']} is in village(city/place) {row['Village']} "
                        f"has a cattle(cow/ox) count of {row['Cattle count']} is registered with Reg. id. of {row['Registration No']} "
                        f"contact no {row['Mobile No']}."
                        f"{'. '.join(goshala_words)}"
                    )
                    line_doc = Document(
                        page_content=normalize(line),
                        metadata={
                            "page_label": page_num + 1,
                            "type": "text_line",
                            "source": pdf_path
                        }
                    )
                    extracted_docs.append(line_doc)
            
            # Extract Regular Text (ignoring tables ideally, but keeping simple here)
            text = page.extract_text()
            normalized_text = normalize(text) if text else ""
            if text:
                doc = Document(
                    page_content=normalized_text,
                    metadata={
                        "page_label": page_num + 1,
                        "type": "text",
                        "source": pdf_path
                    }
                )
                extracted_docs.append(doc)

    # return extracted_docs