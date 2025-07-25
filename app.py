# OncoKB Variant Querier - Streamlit App
# To run this app locally:
# 1. Follow the installation steps for Tesseract-OCR from the previous version.
# 2. Install the required Python libraries:
#    pip install streamlit streamlit-option-menu requests pandas pdfplumber pytesseract Pillow llama-parse
# 3. Save this code as a Python file (e.g., app.py).
# 4. Run it from your terminal:
#    streamlit run app.py

# To deploy on Streamlit Community Cloud:
# 1. Create a GitHub repository with this app.py file.
# 2. Add your API keys as secrets in your app's settings on Streamlit Cloud:
#    - ONCOKB_API_KEY = "your-oncokb-key"
#    - GEMINI_API_KEY = "your-google-ai-key"
#    - LLAMA_CLOUD_API_KEY = "your-llamaclou-key"
# 3. Create a file named 'requirements.txt' in the repository with the following content:
#    requests
#    pandas
#    pdfplumber
#    pytesseract
#    Pillow
#    streamlit
#    streamlit-option-menu
#    llama-parse
#
# 4. Create a file named 'packages.txt' in the repository with the following content:
#    tesseract-ocr
#
# 5. Deploy the app from your Streamlit Community Cloud account, pointing to your GitHub repository.

import streamlit as st
import requests
import pandas as pd
import io
import re
import json
from urllib.parse import quote
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
import os

# Gracefully handle ImportError if llama-parse is not installed correctly
try:
    from llama_parse import LlamaParse
    llama_parse_available = True
except ImportError:
    LlamaParse = None
    llama_parse_available = False


# --- Page Configuration ---
st.set_page_config(
    page_title="NGS Report Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
def get_level_class(level):
    """Returns a color string for Streamlit based on the OncoKB level."""
    level_map = {
        'LEVEL_1': 'green', 'LEVEL_2': 'blue', 'LEVEL_3A': 'orange',
        'LEVEL_3B': 'orange', 'LEVEL_4': 'violet', 'LEVEL_R1': 'red', 'LEVEL_R2': 'red'
    }
    return level_map.get(level, 'gray')

# --- Data Parsing Functions ---
def parse_nccn_text(text):
    """Parses the NCCN text content into a dictionary."""
    if not text:
        return {}
    
    nccn_data = {}
    gene_blocks = re.split(r'\n(?=\s*[A-Z0-9]{2,10}\s*\n)', text)
    
    for block in gene_blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.split('\n')
        if lines and lines[0].strip():
            gene_name = lines[0].strip().upper()
            gene_name = re.sub(r'[^A-Z0-9]', '', gene_name)
            
            if gene_name:
                nccn_data[gene_name] = block
            
    return nccn_data


def extract_variants_from_text(text):
    """Helper function to extract variants from a block of text."""
    variants = []
    alteration_re = re.compile(r'(p\.[A-Za-z0-9*?fs>_]+)', re.IGNORECASE)
    
    for line in text.split('\n'):
        line = line.strip()
        
        line_match = re.search(r'\b([A-Z0-9]{2,10})\b\s+.*variant\s+(.*)', line, re.IGNORECASE)
        
        if line_match:
            gene = line_match.group(1)
            details = line_match.group(2)
            
            alteration_match = alteration_re.search(details)
            if alteration_match:
                alteration = alteration_match.group(1)
                
                if gene.upper() not in ['GENE', 'TYPE', 'DETAILS']:
                    variants.append({'Gene': gene, 'Alteration': alteration})
    return variants


def parse_molecular_report(uploaded_file, llama_api_key):
    """
    Parses the uploaded molecular report file.
    Returns a tuple: (DataFrame, debug_log, image_for_review).
    """
    if uploaded_file is None:
        return (pd.read_csv(io.StringIO(DEFAULT_VARIANTS_CSV)), None, None)

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    image_for_review = None
    
    try:
        if 'csv' in filename or 'xls' in filename:
            df = pd.read_excel(file_bytes) if 'xls' in filename else pd.read_csv(io.BytesIO(file_bytes))
            return (df, None, None)
        
        elif 'pdf' in filename or any(ext in filename for ext in ['.png', '.jpg', '.jpeg']):
            variants = []
            debug_log = "--- PARSING LOG ---\n"
            full_extracted_text = ""
            
            if 'pdf' in filename:
                debug_log += "Attempting Method 1: Standard OCR on PDF...\n"
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        if i == 0:
                            image_for_review = page.to_image(resolution=300).original

                        page_text = page.extract_text()
                        if not page_text or page_text.strip() == "":
                            try:
                                img = page.to_image(resolution=300).original
                                img = img.convert('L')
                                enhancer = ImageEnhance.Contrast(img)
                                img = enhancer.enhance(2)
                                page_text = pytesseract.image_to_string(img)
                            except Exception:
                                page_text = "" 
                        full_extracted_text += page_text + "\n\n"
                variants.extend(extract_variants_from_text(full_extracted_text))
            
            elif any(ext in filename for ext in ['.png', '.jpg', '.jpeg']):
                 debug_log += "Attempting OCR on Image...\n"
                 img = Image.open(io.BytesIO(file_bytes))
                 image_for_review = img
                 img = img.convert('L')
                 enhancer = ImageEnhance.Contrast(img)
                 img = enhancer.enhance(2)
                 full_extracted_text = pytesseract.image_to_string(img)
                 variants.extend(extract_variants_from_text(full_extracted_text))


            if not variants and llama_api_key and llama_parse_available and 'pdf' in filename:
                debug_log += "\nMethod 1 failed. Attempting Method 2: LlamaParse...\n"
                try:
                    with open("temp_report.pdf", "wb") as f:
                        f.write(file_bytes)
                    
                    parser = LlamaParse(api_key=llama_api_key, result_type="markdown")
                    documents = parser.load_data("temp_report.pdf")
                    llama_text = documents[0].text
                    debug_log += f"LlamaParse successfully extracted {len(llama_text)} characters.\n"
                    full_extracted_text = llama_text
                    
                    variants.extend(extract_variants_from_text(llama_text))
                    os.remove("temp_report.pdf")
                except Exception as e:
                    debug_log += f"LlamaParse failed with error: {e}\n"
                    if os.path.exists("temp_report.pdf"):
                        os.remove("temp_report.pdf")

            if variants:
                df = pd.DataFrame(variants).drop_duplicates().reset_index(drop=True)
                return (df, None, image_for_review)
            else:
                debug_log += "\n--- FULL EXTRACTED TEXT (from last attempt) ---\n" + (full_extracted_text or "No text was extracted.")
                return (pd.DataFrame(), debug_log, image_for_review)

    except Exception as e:
        return (None, f"A critical error occurred: {e}", None)

# --- API Call Functions ---
@st.cache_data
def get_oncokb_data(hugo_symbol, alteration, tumor_type, api_token):
    """
    Fetches data from the OncoKB API for a single variant.
    Returns a tuple of (response_json, api_url) for debugging.
    """
    api_alteration = alteration
    
    if api_alteration.lower() == 'p.?':
        api_alteration = 'X1000_splice'
    elif isinstance(api_alteration, str) and api_alteration.lower().startswith('p.'):
        api_alteration = api_alteration[2:]

    base_url = 'https://www.oncokb.org/api/v1' if api_token else 'https://demo.oncokb.org/api/v1'
    api_url = f"{base_url}/annotate/mutations/byProteinChange?hugoSymbol={hugo_symbol}&alteration={api_alteration}"
    if tumor_type:
        api_url += f"&tumorType={tumor_type}"
    
    headers = {'Accept': 'application/json'}
    if api_token:
        headers['Authorization'] = f'Bearer {api_token}'

    try:
        response = requests.get(api_url, headers=headers, verify=False)
        response.raise_for_status()
        return (response.json(), api_url)
    except requests.exceptions.HTTPError as e:
        try:
            return (e.response.json(), api_url)
        except ValueError:
            return ({'error': f"API Error: Status {e.response.status_code}"}, api_url)
    except requests.exceptions.RequestException as e:
        return ({'error': f"Network Error: {e}"}, api_url)

def generate_narrative_summary(all_oncokb_data, nccn_data, tumor_type, gemini_api_key):
    """Generates a narrative summary using the Gemini API."""
    
    prompt_context = "You are a clinical assistant summarizing a molecular report.\n\n"
    prompt_context += f"**Patient Context:** The patient has a tumor of type: {tumor_type or 'Not Specified'}.\n\n"
    prompt_context += "**Source Data:**\n"
    
    prompt_context += "--- OncoKB Information ---\n"
    for data, url in all_oncokb_data: # Unpack the tuple
        if 'query' in data and data.get('geneExist', False):
            prompt_context += json.dumps(data, indent=2) + "\n\n"
            
    prompt_context += "\n--- NCCN Guideline Information ---\n"
    for gene, info in nccn_data.items():
        prompt_context += f"**Gene: {gene}**\n{info}\n\n"
        
    prompt_task = (
        "**Task:** Based *only* on the source data provided above, generate a concise narrative summary "
        "suitable for a clinical report. Structure the summary as follows:\n"
        "1.  **Overall Summary:** A brief, one-paragraph overview of the key findings.\n"
        "2.  **Variant Details:** For each significant variant, provide a bullet point that synthesizes its "
        "prognostic significance and therapeutic implications from both the OncoKB and NCCN data.\n"
        "Do not include information not present in the provided text. Be objective and clinical in tone."
    )
    
    full_prompt = prompt_context + prompt_task
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error generating summary: {e}\n\nResponse from server:\n{response.text}"


# --- UI Rendering Functions ---
def display_oncokb_results(data_tuple, hugo_symbol, alteration):
    """Displays the formatted OncoKB results in a tab."""
    
    data, api_url = data_tuple # Unpack the tuple
    
    st.markdown("**API Query Sent:**")
    st.code(api_url, language="text")
    st.divider()

    if 'error' in data:
        st.error(data['error'])
    elif data.get('query', {}).get('variant') == "UNKNOWN":
        st.warning("Variant not found in OncoKB.")
    else:
        query = data.get('query', {})
        
        # **FIX:** Use the cleaned alteration from the query for the link
        link_alteration = query.get('alteration', alteration)
        oncokb_link = f"https://www.oncokb.org/gene/{query.get('hugoSymbol', hugo_symbol)}/{link_alteration}"
        
        st.markdown(f"**Tumor Type in Query:** `{query.get('tumorType', 'Not specified')}`")
        st.link_button("View on OncoKB", oncokb_link)
        
        st.subheader("Summaries")
        st.info(f"**Gene Summary:** {data.get('geneSummary', 'N/A')}")
        st.info(f"**Variant Summary:** {data.get('variantSummary', 'N/A')}")

        if data.get('treatments'):
            st.subheader("Therapeutic Implications")
            for treatment in data.get('treatments', []):
                drugs = ", ".join([d['drugName'] for d in treatment['drugs']])
                level = treatment['level'].replace('_', ' ')
                indication = treatment.get('indication', {}).get('name', 'N/A')
                pmids = ", ".join([f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})" for pmid in treatment.get('pmids', [])])
                
                # **FIX:** Correctly extract the tumor type name for the treatment level
                cancer_type_info = treatment.get('levelAssociatedCancerType')
                cancer_type_name = cancer_type_info.get('name') if cancer_type_info else None
                
                st.markdown(f"**{drugs}** - `{indication}`")
                if cancer_type_name:
                    st.markdown(f"**For Tumor Type(s):** {cancer_type_name}")
                st.markdown(f"> :{get_level_class(treatment['level'])}[{level}] - {pmids}")

    st.divider()
    with st.expander("Show Raw OncoKB Response"):
        st.json(data)

# --- Main App ---
st.title("NGS Report Assistant")

# --- Sidebar Inputs ---
st.sidebar.header("1. Upload Files")
report_file = st.sidebar.file_uploader("Molecular Report", type=['pdf', 'csv', 'xlsx', 'png', 'jpg', 'jpeg'])

nccn_github_url = st.sidebar.text_input(
    "NCCN File GitHub URL", 
    value="https://raw.githubusercontent.com/Eitan177/ngs_report_assistant/refs/heads/main/nccn_cleaned.txt"
)
nccn_file_upload = st.sidebar.file_uploader("NCCN Information File (Fallback)", type=['txt'])
st.sidebar.info("Provide a GitHub URL for the NCCN file (preferred) or upload it directly.")


st.sidebar.header("2. Query Options")
tumor_type = st.sidebar.text_input("Tumor Type (Applied to all variants)", placeholder="e.g., Melanoma")

st.sidebar.divider()
oncokb_api_token = st.secrets.get("ONCOKB_API_KEY")
if oncokb_api_token:
    st.sidebar.success("OncoKB API key found.")
else:
    st.sidebar.warning("OncoKB API key not found. Using public API.")

gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if gemini_api_key:
    st.sidebar.success("Gemini API key found.")
else:
    st.sidebar.warning("Gemini API key not found. Summary generation will be disabled.")

if llama_parse_available:
    llama_api_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
    if llama_api_key:
        st.sidebar.success("LlamaParse API key found.")
    else:
        st.sidebar.warning("LlamaParse API key not found. Advanced PDF parsing will be disabled.")
else:
    st.sidebar.error("LlamaParse library not found. Advanced PDF parsing is disabled.")
st.sidebar.divider()

# --- Initialize Session State ---
if 'step' not in st.session_state:
    st.session_state.step = "initial"
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'narrative_summary' not in st.session_state:
    st.session_state.narrative_summary = ""
if 'all_oncokb_data' not in st.session_state:
    st.session_state.all_oncokb_data = []

def process_dataframe(df):
    """The main logic for processing a finalized DataFrame."""
    st.session_state.results_df = df
    st.session_state.step = "results"
    
    with st.spinner("Querying OncoKB for all variants..."):
        all_oncokb_data = []
        for index, row in df.iterrows():
            data_tuple = get_oncokb_data(row['Gene'], row['Alteration'], st.session_state.tumor_type, oncokb_api_token)
            all_oncokb_data.append(data_tuple)
        st.session_state.all_oncokb_data = all_oncokb_data


# --- Main Processing Logic ---
if st.sidebar.button("Process Variants", type="primary"):
    st.session_state.narrative_summary = ""
    st.session_state.results_df = pd.DataFrame()
    st.session_state.step = "initial"
    
    if report_file is None:
        st.warning("Please upload a molecular report file.")
    else:
        nccn_text = ""
        if nccn_github_url:
            try:
                response = requests.get(nccn_github_url)
                response.raise_for_status()
                nccn_text = response.text
                st.sidebar.info("Loaded NCCN data from GitHub URL.")
            except Exception as e:
                st.sidebar.error(f"Failed to load from URL: {e}")
        elif nccn_file_upload:
            nccn_text = nccn_file_upload.getvalue().decode('utf-8-sig')
            st.sidebar.info("Loaded NCCN data from uploaded file.")
        
        st.session_state.nccn_data = parse_nccn_text(nccn_text)
        st.session_state.tumor_type = tumor_type
        
        parsing_result = parse_molecular_report(report_file, llama_api_key if llama_parse_available else None)
        if parsing_result is None:
            st.error("The file parser returned an unexpected error.")
        else:
            df, debug_log, image_for_review = parsing_result
            st.session_state.image_for_review = image_for_review
            
            is_tabular = any(ext in report_file.name.lower() for ext in ['.csv', '.xls'])
            
            if df is not None and not df.empty:
                st.session_state.raw_df = df
                if is_tabular and ('Gene' not in df.columns or 'Alteration' not in df.columns):
                    st.session_state.step = "column_selection"
                else:
                    st.session_state.step = "review"
            else:
                st.error("Could not parse any variants from the molecular report.")
                if debug_log:
                    st.subheader("PDF/File Parsing Debug Log")
                    st.text_area("Log", debug_log, height=300)


# --- UI Display Logic ---
if st.session_state.step == "initial":
    st.info("Upload your files and click 'Process Variants' in the sidebar to begin.")

elif st.session_state.step == "column_selection":
    st.warning("Could not automatically identify 'Gene' and 'Alteration' columns.")
    st.info("Please select the correct columns from your file below:")
    
    df_columns = list(st.session_state.raw_df.columns)
    
    col1, col2 = st.columns(2)
    with col1:
        gene_col = st.selectbox("Select the column containing the **Gene**", df_columns, index=None)
    with col2:
        alteration_col = st.selectbox("Select the column containing the **Alteration**", df_columns, index=None)
        
    if st.button("Confirm Columns"):
        if gene_col and alteration_col:
            temp_df = st.session_state.raw_df.rename(columns={gene_col: 'Gene', alteration_col: 'Alteration'})
            process_dataframe(temp_df[['Gene', 'Alteration']])
            st.rerun()
        else:
            st.error("Please select both a Gene and an Alteration column.")

elif st.session_state.step == "review":
    st.header("Review and Edit Parsed Variants")
    st.info("OCR is not always perfect. Please review the extracted variants below and correct any errors before proceeding.")
    
    if st.session_state.get('image_for_review'):
        st.image(st.session_state.image_for_review, caption="Source Image for Review")
        st.divider()

    edited_df = st.data_editor(st.session_state.raw_df, num_rows="dynamic")
    
    if st.button("Confirm and Analyze Variants", type="primary"):
        process_dataframe(edited_df)
        st.rerun()

elif st.session_state.step == "results":
    df = st.session_state.results_df
    nccn_data = st.session_state.nccn_data
    tumor_type = st.session_state.tumor_type
    all_oncokb_data = st.session_state.all_oncokb_data

    st.sidebar.header("3. Generate Summaries")
    if st.sidebar.button("Generate Narrative Summary", disabled=(not gemini_api_key), key="gemini_summary"):
        with st.spinner("Generating AI-powered summary..."):
            summary = generate_narrative_summary(all_oncokb_data, nccn_data, tumor_type, gemini_api_key)
            st.session_state.narrative_summary = summary

    if st.session_state.narrative_summary:
        st.header("Narrative Summary (Gemini)")
        st.markdown(st.session_state.narrative_summary)
        st.divider()

    st.header("OncoKB Results")
    oncokb_tabs_list = [f"{row['Gene']} {row['Alteration']}" for index, row in df.iterrows()]
    if oncokb_tabs_list:
        oncokb_tabs = st.tabs(oncokb_tabs_list)
        for i, (df_index, row) in enumerate(df.iterrows()):
            with oncokb_tabs[i]:
                display_oncokb_results(all_oncokb_data[i], row['Gene'], row['Alteration'])
    
    if nccn_data:
        st.header("NCCN Information")
        unique_genes = df['Gene'].unique()
        if len(unique_genes) > 0:
            nccn_tabs = st.tabs(list(unique_genes))
            for i, gene in enumerate(unique_genes):
                with nccn_tabs[i]:
                    if tumor_type:
                        st.info(f"Showing general NCCN information for **{gene}**. Please review for information relevant to **{tumor_type}**.")
                    info = nccn_data.get(gene.upper(), f"No NCCN information found for {gene} in the uploaded file.")
                    st.markdown(info)
        else:
            st.warning("No unique genes found in the report to display NCCN info.")

    st.header("AI Aggregator Links")
    if not df.empty:
        ai_tabs_list = [f"{row['Gene']} {row['Alteration']}" for index, row in df.iterrows()]
        if ai_tabs_list:
            ai_tabs = st.tabs(ai_tabs_list)
            for i, (df_index, row) in enumerate(df.iterrows()):
                with ai_tabs[i]:
                    gene = row['Gene']
                    alt = row['Alteration']
                    query_text = f"what is the clinical significance of {gene} {alt}"
                    if tumor_type:
                        query_text += f" in {tumor_type}"
                    
                    perplexity_url = f"https://www.perplexity.ai/search?q={quote(query_text)}"
                    st.info(f"**Generated Query:** `{query_text}`")
                    st.link_button("Ask Perplexity.ai", perplexity_url)
    else:
        st.warning("No variants found to generate AI aggregator links.")
