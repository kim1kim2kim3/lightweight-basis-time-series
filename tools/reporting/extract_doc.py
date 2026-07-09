import os
import requests
import json
import zipfile
import io

os.environ["POLARIS_DATAINSIGHT_API_KEY"] = "datainsight-129786c1b2bcdff8690141277e05acf26f9c2d119eb85aedb62d646b52a4347c"
file_path = "c:/Users/ydh33/Downloads/model_실험_논문용_이거보면됨/양식-논문샘플(워드) (2).doc"

def extract_document(file_path: str, api_key: str) -> dict:
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://datainsight-api.polarisoffice.com/api/v1/datainsight/doc-extract",
            headers={"x-po-di-apikey": api_key},
            files={"file": f}
        )

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    zip_buffer = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_buffer) as z:
        json_files = [name for name in z.namelist() if name.endswith('.json')]
        if json_files:
            with z.open(json_files[0]) as jf:
                return json.load(jf)
    raise Exception("No JSON found in ZIP")

def get_all_text(schema: dict) -> str:
    texts = []
    for page in schema.get("pages", []):
        for el in page.get("elements", []):
            if el["type"] == "text" and el.get("content", {}).get("text"):
                texts.append(el["content"]["text"])
    return "\n".join(texts)

api_key = os.environ["POLARIS_DATAINSIGHT_API_KEY"]
schema = extract_document(file_path, api_key)
print(get_all_text(schema))
