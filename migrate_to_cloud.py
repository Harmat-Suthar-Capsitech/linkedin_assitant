"""
Data Migration Utility: Localhost to Cloud (Redis + Qdrant).
Transfers all local Redis conversation/collection state and Qdrant vector collections
to your Cloud providers (Upstash / Redis Cloud and Qdrant Cloud).

Usage:
  python migrate_to_cloud.py
Or configure environment variables:
  CLOUD_REDIS_URL=rediss://default:xxxx@xxxx.upstash.io:6379
  CLOUD_QDRANT_URL=https://xxxx.aws.cloud.qdrant.io:6333
  CLOUD_QDRANT_API_KEY=xxxx
"""
import os
import sys
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

import redis
from qdrant_client import QdrantClient, models

# -------------------------------------------------------------
# Configuration: Local Sources
# -------------------------------------------------------------
LOCAL_REDIS_HOST = os.getenv("LOCAL_REDIS_HOST", "localhost")
LOCAL_REDIS_PORT = int(os.getenv("LOCAL_REDIS_PORT", 6379))
LOCAL_REDIS_DB = int(os.getenv("LOCAL_REDIS_DB", 0))

LOCAL_QDRANT_URL = os.getenv("LOCAL_QDRANT_URL", "http://localhost:6333")

# -------------------------------------------------------------
# Configuration: Cloud Destinations
# -------------------------------------------------------------
# If CLOUD_REDIS_URL is not set separately, it checks REDIS_URL or REDIS_HOST
CLOUD_REDIS_URL = os.getenv("CLOUD_REDIS_URL") or os.getenv("REDIS_URL")
CLOUD_REDIS_HOST = os.getenv("CLOUD_REDIS_HOST")
CLOUD_REDIS_PORT = int(os.getenv("CLOUD_REDIS_PORT", 6379))
CLOUD_REDIS_PASSWORD = os.getenv("CLOUD_REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD") or os.getenv("REDIS_API_KEY")

CLOUD_QDRANT_URL = os.getenv("CLOUD_QDRANT_URL") or (
    os.getenv("QDRANT_URL") if os.getenv("QDRANT_URL", "").startswith("https://") else None
)
CLOUD_QDRANT_API_KEY = os.getenv("CLOUD_QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")


def get_local_redis() -> redis.Redis:
    return redis.Redis(
        host=LOCAL_REDIS_HOST,
        port=LOCAL_REDIS_PORT,
        db=LOCAL_REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def get_cloud_redis() -> redis.Redis:
    if CLOUD_REDIS_URL:
        return redis.from_url(
            CLOUD_REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=10,
            retry_on_timeout=True,
        )
    if CLOUD_REDIS_HOST and CLOUD_REDIS_HOST != "localhost":
        ssl = CLOUD_REDIS_PORT in (6380, 25061) or "upstash" in CLOUD_REDIS_HOST.lower()
        return redis.Redis(
            host=CLOUD_REDIS_HOST,
            port=CLOUD_REDIS_PORT,
            password=CLOUD_REDIS_PASSWORD,
            ssl=ssl,
            decode_responses=True,
            socket_connect_timeout=10,
        )
    raise ValueError(
        "Cloud Redis target not configured. Set CLOUD_REDIS_URL or REDIS_URL in .env "
        "(e.g. rediss://default:<password>@<host>:<port>)"
    )


def get_local_qdrant() -> QdrantClient:
    return QdrantClient(url=LOCAL_QDRANT_URL, check_compatibility=False)


def get_cloud_qdrant() -> QdrantClient:
    if not CLOUD_QDRANT_URL:
        raise ValueError(
            "Cloud Qdrant URL not configured. Set CLOUD_QDRANT_URL or QDRANT_URL in .env "
            "(e.g. https://<cluster-id>.aws.cloud.qdrant.io:6333)"
        )
    return QdrantClient(
        url=CLOUD_QDRANT_URL,
        api_key=CLOUD_QDRANT_API_KEY,
        check_compatibility=False,
    )


# -------------------------------------------------------------
# Redis Migration
# -------------------------------------------------------------
def migrate_redis():
    print("\n" + "=" * 60)
    print("📦 [1/2] Migrating Redis Data (Local -> Cloud)...")
    print("=" * 60)

    try:
        r_local = get_local_redis()
        local_keys = r_local.keys("linkedin_assistant:*")
        print(f"[*] Found {len(local_keys)} keys matching 'linkedin_assistant:*' on local Redis.")
    except Exception as e:
        print(f"❌ Failed to connect to local Redis: {e}")
        return False

    if not local_keys:
        print("[!] No keys found in local Redis to migrate.")
        return True

    try:
        r_cloud = get_cloud_redis()
        r_cloud.ping()
        print("✅ Successfully connected to Cloud Redis!")
    except Exception as e:
        print(f"❌ Failed to connect to Cloud Redis: {e}")
        return False

    success_count = 0
    pipe = r_cloud.pipeline()
    for idx, key in enumerate(local_keys, start=1):
        try:
            ktype = r_local.type(key)
            ttl = r_local.ttl(key)
            ttl_val = ttl if ttl and ttl > 0 else (30 * 24 * 3600)

            if ktype == "string":
                val = r_local.get(key)
                pipe.set(key, val, ex=ttl_val)
            elif ktype == "set":
                members = r_local.smembers(key)
                if members:
                    pipe.delete(key)
                    pipe.sadd(key, *members)
                    pipe.expire(key, ttl_val)
            elif ktype == "hash":
                hdata = r_local.hgetall(key)
                if hdata:
                    pipe.delete(key)
                    pipe.hset(key, mapping=hdata)
                    pipe.expire(key, ttl_val)
            elif ktype == "list":
                items = r_local.lrange(key, 0, -1)
                if items:
                    pipe.delete(key)
                    pipe.rpush(key, *items)
                    pipe.expire(key, ttl_val)
            success_count += 1
            if idx % 20 == 0:
                pipe.execute()
                print(f"  -> Migrated {idx}/{len(local_keys)} keys...")
        except Exception as err:
            print(f"  [WARN] Failed to copy key {key}: {err}")

    try:
        pipe.execute()
    except Exception as e:
        print(f"  [WARN] Final pipeline execute: {e}")

    print(f"✅ Successfully transferred {success_count}/{len(local_keys)} keys to Cloud Redis!")
    return True


# -------------------------------------------------------------
# Qdrant Migration
# -------------------------------------------------------------
def migrate_qdrant():
    print("\n" + "=" * 60)
    print("🚀 [2/2] Migrating Qdrant Vector Stores (Local -> Cloud)...")
    print("=" * 60)

    try:
        qc_local = get_local_qdrant()
        local_colls = [c.name for c in qc_local.get_collections().collections]
        print(f"[*] Found {len(local_colls)} collections on local Qdrant: {local_colls}")
    except Exception as e:
        print(f"❌ Failed to connect to local Qdrant: {e}")
        return False

    if not local_colls:
        print("[!] No collections found on local Qdrant.")
        return True

    try:
        qc_cloud = get_cloud_qdrant()
        qc_cloud.get_collections()
        print("✅ Successfully connected to Cloud Qdrant!")
    except Exception as e:
        print(f"❌ Failed to connect to Cloud Qdrant: {e}")
        return False

    for coll_name in local_colls:
        try:
            local_info = qc_local.get_collection(coll_name)
            local_count = qc_local.count(coll_name).count
            print(f"\n📂 Processing collection: '{coll_name}' ({local_count} points)...")

            # Recreate on cloud if not exists
            if not qc_cloud.collection_exists(coll_name):
                print(f"  -> Creating collection '{coll_name}' on Cloud Qdrant...")
                qc_cloud.create_collection(
                    collection_name=coll_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=512,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False),
                        )
                    },
                )
                print(f"  ✅ Created '{coll_name}' on Cloud Qdrant.")
            else:
                print(f"  ℹ️ Collection '{coll_name}' already exists on Cloud Qdrant.")

            # Scroll and migrate points
            offset = None
            total_migrated = 0
            batch_size = 50

            while True:
                points_batch, next_offset = qc_local.scroll(
                    collection_name=coll_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                if not points_batch:
                    break

                cloud_points = []
                for pt in points_batch:
                    vec = pt.vector
                    dense_vec = vec.get("dense") if isinstance(vec, dict) else vec
                    sparse_data = vec.get("sparse") if isinstance(vec, dict) else None

                    vectors_dict = {}
                    if dense_vec is not None:
                        vectors_dict["dense"] = dense_vec
                    if sparse_data is not None:
                        if hasattr(sparse_data, "indices"):
                            vectors_dict["sparse"] = models.SparseVector(
                                indices=list(sparse_data.indices),
                                values=list(sparse_data.values),
                            )
                        elif isinstance(sparse_data, dict):
                            vectors_dict["sparse"] = models.SparseVector(
                                indices=sparse_data["indices"],
                                values=sparse_data["values"],
                            )

                    cloud_points.append(
                        models.PointStruct(
                            id=pt.id,
                            vector=vectors_dict,
                            payload=pt.payload,
                        )
                    )

                if cloud_points:
                    qc_cloud.upsert(
                        collection_name=coll_name,
                        points=cloud_points,
                        wait=True,
                    )
                    total_migrated += len(cloud_points)
                    print(f"  -> Transferred {total_migrated}/{local_count} points...")

                offset = next_offset
                if offset is None:
                    break

            cloud_count = qc_cloud.count(coll_name).count
            print(f"✅ Collection '{coll_name}' sync complete! (Cloud has {cloud_count} points)")

        except Exception as e:
            print(f"❌ Error migrating collection '{coll_name}': {e}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LinkedIn Assistant - Cloud Migration Tool")
    print("=" * 60)

    redis_ok = False
    try:
        redis_ok = migrate_redis()
    except Exception as e:
        print(f"Redis migration note: {e}")

    qdrant_ok = False
    try:
        qdrant_ok = migrate_qdrant()
    except Exception as e:
        print(f"Qdrant migration note: {e}")

    print("\n" + "=" * 60)
    print("🏁 Migration Run Summary:")
    print(f"  - Redis Migration : {'✅ Complete' if redis_ok else '⚠️ Needs valid Cloud Redis credentials in .env'}")
    print(f"  - Qdrant Migration: {'✅ Complete' if qdrant_ok else '⚠️ Needs valid Cloud Qdrant credentials in .env'}")
    print("=" * 60 + "\n")
