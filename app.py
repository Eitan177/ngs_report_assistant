# OncoKB Variant Querier - Streamlit App
# To run this app locally:
# 1. Follow the installation steps for Tesseract-OCR from the previous version.
# 2. Install the required Python libraries:
#    pip install streamlit streamlit-option-menu requests pandas pdfplumber pytesseract Pillow
# 3. Save this code as a Python file (e.g., app.py).
# 4. Run it from your terminal:
#    streamlit run app.py

# To deploy on Streamlit Community Cloud:
# 1. Create a GitHub repository with this app.py file.
# 2. Add your API keys as secrets in your app's settings on Streamlit Cloud:
#    - ONCOKB_API_KEY = "your-oncokb-key"
#    - GEMINI_API_KEY = "your-google-ai-key"
# 3. Create a file named 'requirements.txt' in the repository with the following content:
#    requests
#    pandas
#    pdfplumber
#    pytesseract
#    Pillow
#    streamlit
#    streamlit-option-menu
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
def parse_nccn_file(uploaded_file):
    """Parses the uploaded NCCN text file with a more robust strategy."""
    if uploaded_file is None:
        return {}
    
    text = uploaded_file.getvalue().decode('utf-8-sig')
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


def parse_molecular_report(uploaded_file):
    """
    Parses the uploaded molecular report file.
    Returns a tuple: (DataFrame, debug_log).
    """
    if uploaded_file is None:
        # If no file is uploaded, use the default example data.
        return (pd.read_csv(io.StringIO(DEFAULT_VARIANTS_CSV)), None)

    filename = uploaded_file.name
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    
    try:
        if 'csv' in filename:
            return (pd.read_csv(file_bytes), None)
        elif 'xls' in filename:
            return (pd.read_excel(file_bytes), None)
        elif 'pdf' in filename:
            variants = []
            debug_log = "--- PDF PARSING LOG ---\n"
            full_ocr_text = ""
            
            line_finder_re = re.compile(r'\b([A-Z0-9]{2,10})\b\s+(?:Frameshift|Missense)\s+variant\s+(?:Details\s+)?([^\s,]+)')

            with pdfplumber.open(file_bytes) as pdf:
                debug_log += f"Successfully opened PDF. Found {len(pdf.pages)} pages.\n"
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    
                    if not page_text or page_text.strip() == "":
                        debug_log += f"Page {i+1}: No text extracted. Attempting OCR...\n"
                        try:
                            img = page.to_image(resolution=300).original
                            img = img.convert('L')
                            enhancer = ImageEnhance.Contrast(img)
                            img = enhancer.enhance(2)
                            
                            ocr_text = pytesseract.image_to_string(img)
                            page_text = ocr_text
                            full_ocr_text += ocr_text + "\n\n"
                            debug_log += f"Page {i+1}: OCR extracted {len(page_text)} characters.\n"
                        except Exception as ocr_error:
                            debug_log += f"Page {i+1}: OCR failed. Tesseract may not be installed correctly. Error: {ocr_error}\n"
                            continue
                    else:
                        debug_log += f"Page {i+1}: Standard text extraction found {len(page_text)} characters.\n"

                    for line in page_text.split('\n'):
                        match = line_finder_re.search(line)
                        if match:
                            gene = match.group(1)
                            alteration = match.group(2)
                            
                            if 'p.' in alteration.lower():
                                alteration = re.sub(r'.*p\.', '', alteration, flags=re.IGNORECASE)
                            
                            variants.append({'Gene': gene, 'Alteration': alteration})

            if variants:
                df = pd.DataFrame(variants).drop_duplicates().reset_index(drop=True)
                return (df, None)
            else:
                debug_log += "\n--- FULL EXTRACTED TEXT ---\n" + (full_ocr_text or "No text was extracted from any page.")
                return (pd.DataFrame(), debug_log)

    except Exception as e:
        error_message = f"A critical error occurred during file parsing: {e}"
        return (None, error_message)

# --- API Call Functions ---
@st.cache_data
def get_oncokb_data(hugo_symbol, alteration, tumor_type, api_token):
    """Fetches data from the OncoKB API for a single variant."""
    api_alteration = alteration
    if isinstance(api_alteration, str) and api_alteration.startswith('p.'):
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
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            return e.response.json()
        except ValueError:
            return {'error': f"API Error: Status {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {'error': f"Network Error: {e}"}

def generate_narrative_summary(all_oncokb_data, nccn_data, tumor_type, gemini_api_key):
    """Generates a narrative summary using the Gemini API."""
    
    prompt_context = "You are a clinical assistant summarizing a molecular report.\n\n"
    prompt_context += f"**Patient Context:** The patient has a tumor of type: {tumor_type or 'Not Specified'}.\n\n"
    prompt_context += "**Source Data:**\n"
    
    # Add OncoKB data to the prompt
    prompt_context += "--- OncoKB Information ---\n"
    for variant_data in all_oncokb_data:
        # Only include successful queries in the prompt
        if 'query' in variant_data and variant_data.get('geneExist', False):
            prompt_context += json.dumps(variant_data, indent=2) + "\n\n"
            
    # Add NCCN data to the prompt
    prompt_context += "\n--- NCCN Guideline Information ---\n"
    for gene, info in nccn_data.items():
        prompt_context += f"**Gene: {gene}**\n{info}\n\n"
        
    # Define the task for the model
    prompt_task = (
        "**Task:** Based *only* on the source data provided above, generate a concise narrative summary "
        "suitable for a clinical report. Structure the summary as follows:\n"
        "1.  **Overall Summary:** A brief, one-paragraph overview of the key findings.\n"
        "2.  **Variant Details:** For each significant variant, provide a bullet point that synthesizes its "
        "prognostic significance and therapeutic implications from both the OncoKB and NCCN data.\n"
        "Do not include information not present in the provided text. Be objective and clinical in tone."
    )
    
    full_prompt = prompt_context + prompt_task
    
    # **FIX:** Changed model from 'gemini-pro' to 'gemini-1.5-flash'
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
def display_oncokb_results(data, hugo_symbol, alteration):
    """Displays the formatted OncoKB results in a tab."""
    
    if 'error' in data:
        st.error(data['error'])
    elif data.get('query', {}).get('variant') == "UNKNOWN":
        st.warning("Variant not found in OncoKB.")
    else:
        query = data.get('query', {})
        oncokb_link = f"https://www.oncokb.org/gene/{hugo_symbol}/{alteration}"
        
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
                
                st.markdown(f"**{drugs}** - `{indication}`")
                st.markdown(f"> :{get_level_class(treatment['level'])}[{level}] - {pmids}")

    st.divider()
    with st.expander("Show Raw OncoKB Response"):
        st.json(data)

# --- Main App ---
st.title("NGS Report Assistant")

# --- Sidebar Inputs ---
st.sidebar.header("1. Upload Files")
report_file = st.sidebar.file_uploader("Molecular Report", type=['pdf', 'csv', 'xlsx'])
nccn_file = st.sidebar.file_uploader("NCCN Information File (Optional)", type=['txt'])
st.sidebar.info("Format your NCCN .txt file with the gene name on its own line. You can use '---' to separate entries if you wish.")


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
st.sidebar.divider()

# --- Main Processing Logic ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

if st.sidebar.button("Process Variants", type="primary"):
    if report_file is None:
        st.warning("Please upload a molecular report file.")
    else:
        nccn_data = parse_nccn_file(nccn_file)
        
        parsing_result = parse_molecular_report(report_file)
        if parsing_result is None:
            st.error("The file parser returned an unexpected error.")
        else:
            df, debug_log = parsing_result
            st.session_state.results_df = df
            st.session_state.nccn_data = nccn_data
            st.session_state.tumor_type = tumor_type
            
            if df is None or df.empty:
                st.error("Could not parse any variants from the molecular report.")
                if debug_log:
                    st.subheader("PDF Parsing Debug Log")
                    st.text_area("Log", debug_log, height=300)
            else:
                st.success(f"Successfully parsed {len(df)} variants from the report.")
                # Store data for summary generation later
                all_oncokb_data = []
                for index, row in df.iterrows():
                    data = get_oncokb_data(row['Gene'], row['Alteration'], tumor_type, oncokb_api_token)
                    all_oncokb_data.append(data)
                st.session_state.all_oncokb_data = all_oncokb_data


if not st.session_state.results_df.empty:
    df = st.session_state.results_df
    nccn_data = st.session_state.nccn_data
    tumor_type = st.session_state.tumor_type
    all_oncokb_data = st.session_state.all_oncokb_data

    # --- OncoKB Results Section ---
    st.header("OncoKB Results")
    oncokb_tabs_list = [f"{row['Gene']} p.{row['Alteration']}" for index, row in df.iterrows()]
    if oncokb_tabs_list:
        oncokb_tabs = st.tabs(oncokb_tabs_list)
        for i, row in df.iterrows():
            with oncokb_tabs[i]:
                display_oncokb_results(all_oncokb_data[i], row['Gene'], row['Alteration'])
    
    # --- NCCN Information Section ---
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

    # --- AI Aggregator Section ---
    st.header("AI Aggregator Links")
    ai_tabs_list = [f"{row['Gene']} {row['Alteration']}" for index, row in df.iterrows()]
    if ai_tabs_list:
        ai_tabs = st.tabs(ai_tabs_list)
        for i, row in df.iterrows():
            with ai_tabs[i]:
                gene = row['Gene']
                alt = row['Alteration']
                query_text = f"what is the clinical significance of {gene} p.{alt}"
                if tumor_type:
                    query_text += f" in {tumor_type}"
                
                perplexity_url = f"https://www.perplexity.ai/search?q={quote(query_text)}"
                st.info(f"**Generated Query:** `{query_text}`")
                st.link_button("Ask Perplexity.ai", perplexity_url)
    
    # --- Narrative Summary Section ---
    st.header("Narrative Summary")
    if st.button("Generate Summary", disabled=(not gemini_api_key)):
        with st.spinner("Generating AI-powered summary..."):
            summary = generate_narrative_summary(all_oncokb_data, nccn_data, tumor_type, gemini_api_key)
            st.markdown(summary)

else:
    st.info("Upload your files and click 'Process Variants' in the sidebar to begin.")

# Sample data for initial view if no file is uploaded
DEFAULT_VARIANTS_CSV = """Gene,Alteration
JAK2,V617F
"""
