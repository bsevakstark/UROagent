"""
setup_rag.py
------------
Builds the FAISS vector store from mock AUA clinical guidelines.
Run once before launching the agent: `python setup_rag.py`

Dependencies: langchain, langchain-community, openai, faiss-cpu, tiktoken
"""

import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Mock AUA / NCCN Clinical Guideline Chunks
# In a real deployment these would be parsed from official PDF guidelines.
# ---------------------------------------------------------------------------
GUIDELINE_CHUNKS = [
    # PSA Screening
    Document(
        page_content=(
            "PSA Screening (AUA 2023): The AUA recommends shared decision-making "
            "for PSA-based screening in men aged 55–69. For men at average risk, "
            "screening interval may be every 2 years for PSA < 2.5 ng/mL. "
            "Screening is NOT recommended for men under 40 or over 70 with < 10-year "
            "life expectancy."
        ),
        metadata={"source": "AUA_PSA_Screening_2023", "topic": "PSA Screening"},
    ),
    # PSA Thresholds
    Document(
        page_content=(
            "PSA Thresholds (AUA): A PSA level of 4.0 ng/mL is a common threshold "
            "prompting biopsy referral. PSA between 4–10 ng/mL is considered 'gray zone'; "
            "free PSA ratio < 10% increases cancer probability. PSA > 10 ng/mL is highly "
            "suspicious for clinically significant prostate cancer. PSA velocity > 0.75 "
            "ng/mL/year warrants further evaluation."
        ),
        metadata={"source": "AUA_PSA_Thresholds", "topic": "PSA Thresholds"},
    ),
    # Prostate Cancer Staging
    Document(
        page_content=(
            "Prostate Cancer TNM Staging (AJCC 8th Ed): T1 – clinically inapparent tumor; "
            "T1c – tumor identified by needle biopsy. T2 – tumor confined within the prostate; "
            "T2a – one-half of one lobe. T3 – tumor extends through the prostatic capsule; "
            "T3a – extracapsular extension; T3b – seminal vesicle invasion. "
            "T4 – tumor is fixed or invades adjacent structures (bladder, rectum)."
        ),
        metadata={"source": "AJCC_Staging_8th", "topic": "Prostate Cancer Staging"},
    ),
    # Gleason Score
    Document(
        page_content=(
            "Gleason Score / Grade Groups (AUA): Gleason 6 (Grade Group 1) – low risk, "
            "active surveillance often appropriate. Gleason 3+4=7 (Grade Group 2) – "
            "intermediate favorable risk. Gleason 4+3=7 (Grade Group 3) – intermediate "
            "unfavorable risk. Gleason 8 (Grade Group 4) – high risk. "
            "Gleason 9–10 (Grade Group 5) – very high risk, aggressive multimodal therapy indicated."
        ),
        metadata={"source": "AUA_Gleason_GradeGroups", "topic": "Gleason Score"},
    ),
    # Active Surveillance
    Document(
        page_content=(
            "Active Surveillance (AUA 2022): Recommended for very low-risk and low-risk "
            "prostate cancer (Gleason 6, PSA < 10, T1c–T2a). Protocol includes PSA every "
            "6 months, DRE annually, repeat biopsy at 12 months, then every 2–3 years. "
            "MRI is recommended before confirmatory biopsy. Reclassification triggers "
            "include Gleason upgrade or > 2 positive cores."
        ),
        metadata={"source": "AUA_ActiveSurveillance_2022", "topic": "Active Surveillance"},
    ),
    # Radical Prostatectomy
    Document(
        page_content=(
            "Radical Prostatectomy (AUA): Indicated for clinically localized prostate cancer "
            "(T1–T2) with life expectancy > 10 years. Robotic-assisted RP (RARP) is the most "
            "common approach. Nerve-sparing technique preferred to preserve potency. "
            "Lymph node dissection recommended for intermediate/high-risk disease. "
            "Biochemical recurrence defined as PSA ≥ 0.2 ng/mL after surgery."
        ),
        metadata={"source": "AUA_RP_Guidelines", "topic": "Radical Prostatectomy"},
    ),
    # Radiation Therapy
    Document(
        page_content=(
            "Radiation Therapy for Prostate Cancer (AUA/ASTRO): External Beam Radiation "
            "Therapy (EBRT) and brachytherapy are established treatments for localized "
            "prostate cancer. EBRT dose ≥ 75.6 Gy recommended. Hypofractionated EBRT "
            "is a standard alternative. Androgen Deprivation Therapy (ADT) combined with "
            "EBRT improves outcomes in high-risk disease. Brachytherapy monotherapy "
            "appropriate for low-risk, favorable intermediate-risk disease."
        ),
        metadata={"source": "AUA_ASTRO_RT_Guidelines", "topic": "Radiation Therapy"},
    ),
    # Androgen Deprivation Therapy
    Document(
        page_content=(
            "Androgen Deprivation Therapy (ADT) (AUA): ADT (LHRH agonists/antagonists "
            "or bilateral orchiectomy) is the foundation for metastatic prostate cancer. "
            "ADT + novel hormonal agents (enzalutamide, abiraterone) recommended for "
            "metastatic castration-sensitive disease. Side effects include hot flashes, "
            "osteoporosis, metabolic syndrome, and cardiovascular events. Bone-protective "
            "agents (denosumab, zoledronic acid) should be considered."
        ),
        metadata={"source": "AUA_ADT_Guidelines", "topic": "ADT"},
    ),
    # Renal Cell Carcinoma
    Document(
        page_content=(
            "Renal Cell Carcinoma Management (AUA 2022): T1a RCC (< 4 cm) – partial "
            "nephrectomy is preferred over radical nephrectomy. Active surveillance or "
            "thermal ablation are alternatives for elderly/comorbid patients. T1b–T2 – "
            "partial nephrectomy if technically feasible. T3–T4 or node-positive – "
            "radical nephrectomy with lymph node dissection. Adjuvant pembrolizumab "
            "recommended for high-risk resected RCC."
        ),
        metadata={"source": "AUA_RCC_2022", "topic": "Renal Cell Carcinoma"},
    ),
    # Bladder Cancer
    Document(
        page_content=(
            "Bladder Cancer (AUA/SUO 2023): Non-muscle invasive bladder cancer (NMIBC): "
            "TURBT + intravesical BCG for high-risk NMIBC. Muscle-invasive bladder cancer "
            "(MIBC, T2+): neoadjuvant cisplatin-based chemotherapy followed by radical "
            "cystectomy is the standard of care. Urinary diversion (ileal conduit or "
            "neobladder) required post-cystectomy. Enhanced cystoscopy (blue-light) "
            "improves NMIBC detection."
        ),
        metadata={"source": "AUA_Bladder_2023", "topic": "Bladder Cancer"},
    ),
    # BPH
    Document(
        page_content=(
            "Benign Prostatic Hyperplasia (BPH) Management (AUA 2022): First-line: "
            "alpha-1 blockers (tamsulosin, alfuzosin) for symptom relief. 5-alpha reductase "
            "inhibitors (finasteride, dutasteride) for large prostates (> 30 mL) to reduce "
            "volume and progression risk. Combination therapy for severe/progressive symptoms. "
            "Surgical options: TURP (gold standard), laser procedures (HoLEP, GreenLight), "
            "or minimally invasive therapies (UroLift, Rezum) for appropriate candidates."
        ),
        metadata={"source": "AUA_BPH_2022", "topic": "BPH"},
    ),
    # Testicular Cancer
    Document(
        page_content=(
            "Testicular Germ Cell Tumors (AUA): Radical inguinal orchiectomy is the initial "
            "diagnostic and therapeutic procedure. Seminoma Stage I: active surveillance "
            "preferred over adjuvant therapy for low-risk. Non-seminoma Stage I: surveillance "
            "or retroperitoneal lymph node dissection (RPLND). Metastatic disease: BEP "
            "chemotherapy (bleomycin, etoposide, cisplatin) is standard. Tumor markers: "
            "AFP, hCG, LDH are essential for staging and monitoring."
        ),
        metadata={"source": "AUA_TesticularCancer", "topic": "Testicular Cancer"},
    ),
]


def build_vector_store():
    """Embed guideline chunks and persist a FAISS index to disk."""
    os.makedirs("data", exist_ok=True)
    print("🔨 Building FAISS vector store from AUA guidelines...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(GUIDELINE_CHUNKS, embeddings)
    vectorstore.save_local("data/aua_guidelines_index")
    print(f"✅ Vector store saved to data/aua_guidelines_index")
    print(f"   {len(GUIDELINE_CHUNKS)} guideline chunks indexed.")


if __name__ == "__main__":
    build_vector_store()
