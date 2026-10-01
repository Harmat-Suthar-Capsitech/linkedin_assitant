from pathlib import Path
import hashlib

import ollama
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

EXCEL_FILE = Path(
    r"C:\Users\CT_USER\OneDrive\Documents\CIT\Internship work"
    r"\linkedin_assitant\test_vector\jobs_output.xlsx"
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = "linkedin_jobs"

# Ollama embedding model
EMBEDDING_MODEL = "embeddinggemma:300m"

# Sparse model
SPARSE_MODEL = "Qdrant/bm25"

# Current dataset is only ~200-250 jobs.
# No need for very large batches.
BATCH_SIZE = 64

TENANT_ID = "global"


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """Make Excel values safe for Qdrant payloads."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def make_job_id(row: pd.Series, row_number: int) -> str:
    """Create a stable ID for each job."""

    source_url = clean_value(row.get("Source URL"))

    if source_url:
        raw = str(source_url).strip().lower()
    else:
        title = str(
            clean_value(row.get("Job Title")) or ""
        ).strip().lower()

        company = str(
            clean_value(row.get("Company")) or ""
        ).strip().lower()

        location = str(
            clean_value(row.get("Location")) or ""
        ).strip().lower()

        raw = f"{title}|{company}|{location}"

    if not raw:
        raw = f"row-{row_number}"

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


def parse_experience_years(val) -> float | None:
    if pd.isna(val) or val is None:
        return None
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_salary(val) -> float | None:
    if pd.isna(val) or val is None:
        return None
    import re
    s = str(val).replace(",", "").strip()
    m = re.search(r"\$?(\d+(?:\.\d+)?)\s*(k|k/year|/yr|/year)?", s, re.IGNORECASE)
    if m:
        try:
            num = float(m.group(1))
            unit = m.group(2)
            if unit and "k" in unit.lower():
                num *= 1000
            return num
        except ValueError:
            return None
    return None


def build_payload(
    row: pd.Series,
    tenant_id: str,
    job_id: str,
) -> dict:
    """Store all Excel fields as Qdrant payload along with normalized filter fields."""

    payload = {}

    for column, value in row.items():
        payload[column] = clean_value(value)

    payload["tenant_id"] = tenant_id
    payload["job_id"] = job_id

    # Normalized fields matching Qdrant payload indexes and filter queries
    payload["job_Title"] = clean_value(row.get("Job Title"))
    payload["job_title"] = clean_value(row.get("Job Title"))
    payload["company"] = clean_value(row.get("Company"))
    payload["industry"] = clean_value(row.get("Industry"))
    payload["company_size"] = clean_value(row.get("Company Size"))
    payload["location"] = clean_value(row.get("Location"))
    payload["remote_hybrid"] = clean_value(row.get("Remote/Hybrid"))
    payload["wfh"] = clean_value(row.get("WFH"))
    payload["onsite"] = clean_value(row.get("Onsite"))
    payload["experience_level"] = clean_value(row.get("Experience Level"))
    payload["job_category"] = clean_value(row.get("Job Category"))
    payload["verified_status"] = clean_value(row.get("Verified Status"))
    payload["skills_required"] = clean_value(row.get("Skills Required"))
    payload["job_summary"] = clean_value(row.get("Job Summary"))
    payload["source_url"] = clean_value(row.get("Source URL"))
    payload["experience_years"] = parse_experience_years(row.get("Experience (Years)"))
    payload["salary_min"] = parse_salary(row.get("Salary Min"))
    payload["salary_max"] = parse_salary(row.get("Salary Max"))

    return payload



def valid_job(row: pd.Series) -> bool:
    """Skip invalid/search rows."""

    title = clean_value(row.get("Job Title"))
    job_summary = clean_value(
        row.get("Job Summary")
    )

    if not title or not job_summary:
        return False

    if str(title).lstrip().startswith("🔍 Search:"):
        return False

    return True


# ============================================================
# OLLAMA EMBEDDING
# ============================================================

def create_dense_embeddings(texts):
    """
    Generate dense embeddings using Ollama EmbeddingGemma.
    """

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = response["embeddings"]

    return embeddings


# ============================================================
# QDRANT
# ============================================================

def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    client = QdrantClient(
        url=url,
        api_key=api_key,
        check_compatibility=False,
    )

    # Test connection
    client.get_collections()

    print(
        f"[OK] Connected to Qdrant: {url}"
    )

    return client


def create_collection(
    client: QdrantClient,
    dense_dimension: int,
):

    if client.collection_exists(
        COLLECTION_NAME
    ):
        print(
            f"[INFO] Collection already exists: "
            f"{COLLECTION_NAME}"
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config={
            "dense": models.VectorParams(
                size=dense_dimension,
                distance=models.Distance.COSINE,
            )
        },

        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )

    print(
        f"[OK] Created collection: "
        f"{COLLECTION_NAME}"
    )


# ============================================================
# PAYLOAD INDEXES
# ============================================================

def create_filter_indexes(
    client: QdrantClient,
):

    keyword_fields = [
        "tenant_id",
        "job_Title",
        "company",
        "industry",
        "company_size",
        "location",
        "remote_hybrid",
        "wfh",
        "onsite",
        "experience_level",
        "job_category",
        "verified_status",
    ]

    for field in keyword_fields:

        try:

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.TEXT,
            )

            print(
                f"[INDEX] Created keyword index: "
                f"{field}"
            )

        except Exception as exc:

            print(
                f"[WARN] Index skipped for "
                f"'{field}': {exc}"
            )

    numeric_fields = [
        "salary_min",
        "salary_max",
        "experience_years",
    ]

    for field in numeric_fields:

        try:

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.FLOAT,
            )

            print(
                f"[INDEX] Created numeric index: "
                f"{field}"
            )

        except Exception as exc:

            print(
                f"[WARN] Numeric index skipped "
                f"for '{field}': {exc}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("LinkedIn Assistant - Qdrant Job Storage")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 1. Check Excel
    # --------------------------------------------------------

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"Excel file not found:\n"
            f"{EXCEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load Excel
    # --------------------------------------------------------

    print("[INFO] Loading jobs...")

    df = pd.read_excel(
        EXCEL_FILE
    )

    print(
        f"[INFO] Total rows: {len(df)}"
    )

    df = df[
        df.apply(
            valid_job,
            axis=1
        )
    ].reset_index(drop=True)

    print(
        f"[INFO] Valid jobs: {len(df)}"
    )

    if len(df) == 0:
        raise ValueError(
            "No valid jobs found."
        )

    # --------------------------------------------------------
    # 3. Load Sparse Model
    # --------------------------------------------------------

    print(
        "[INFO] Loading BM25 sparse model..."
    )

    sparse_model = SparseTextEmbedding(
        model_name=SPARSE_MODEL
    )

    # --------------------------------------------------------
    # 4. Qdrant
    # --------------------------------------------------------

    client = get_client()

    # --------------------------------------------------------
    # 5. Determine Embedding Dimension
    # --------------------------------------------------------

    # print(
    #     "[INFO] Checking EmbeddingGemma dimension..."
    # )

    # test_embedding = create_dense_embeddings(
    #     ["test"]
    # )[0]
    # test_embedding_512 = test_embedding[:512]
    # dense_dimension = 512

    # print(
    #     f"[INFO] Embedding dimension: "
    #     f"{dense_dimension}"
    # )

    # --------------------------------------------------------
    # 6. Create Collection
    # --------------------------------------------------------

    create_collection(
        client,
        512
    )

    create_filter_indexes(
        client
    )

    # --------------------------------------------------------
    # 7. Process Jobs
    # --------------------------------------------------------

    for start in range(
        0,
        len(df),
        BATCH_SIZE
    ):

        batch = df.iloc[
            start:start + BATCH_SIZE
        ]

        texts = [
            str(
                clean_value(
                    row["Embedding Text"]
                )
            ).strip()

            for _, row in batch.iterrows()
        ]

        print()
        print(
            f"[INFO] Processing jobs "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"/ {len(df)}"
        )

        # ----------------------------------------------------
        # Dense Embedding - Ollama
        # ----------------------------------------------------

        dense_vectors = (
            create_dense_embeddings(
                texts
            )
        )
        dense_vectors = [
            vector[:512]
            for vector in dense_vectors
        ]

        # ----------------------------------------------------
        # Sparse Embedding - BM25
        # ----------------------------------------------------

        sparse_vectors = list(
            sparse_model.embed(
                texts
            )
        )

        # ----------------------------------------------------
        # Create Qdrant Points
        # ----------------------------------------------------

        points = []

        for (
            row_index,
            ((_, row), dense, sparse),
        ) in enumerate(
            zip(
                batch.iterrows(),
                dense_vectors,
                sparse_vectors,
            ),
            start=start,
        ):

            job_id = make_job_id(
                row,
                row_index,
            )

            payload = build_payload(
                row,
                TENANT_ID,
                job_id,
            )

            point = models.PointStruct(
                id=job_id,

                vector={
                    "dense": dense,

                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },

                payload=payload,
            )

            points.append(point)

        # ----------------------------------------------------
        # Store in Qdrant
        # ----------------------------------------------------

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"[OK] Stored {len(points)} jobs"
        )

    # --------------------------------------------------------
    # 8. Verify
    # --------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Points     : {info.points_count}"
    )

    # print(
    #     f"Dense      : {dense_dimension}D"
    # )

    print(
        f"Sparse     : BM25"
    )

    print(
        f"Tenant     : {TENANT_ID}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()