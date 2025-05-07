# lambda/index.py

import json
import os
import re
import urllib.request
import urllib.error
import datetime
import hashlib
import hmac

# ARN からリージョンを抽出
def extract_region_from_arn(arn):
    match = re.search(r'arn:aws:lambda:([^:]+):', arn)
    return match.group(1) if match else "us-east-1"

# SigV4 用 HMAC-SHA256
def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

# 署名鍵の生成
def get_signature_key(key, date_stamp, region, service):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing

# 環境変数からモデルIDを取得
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

def lambda_handler(event, context):
    try:
        # 実行リージョンとエンドポイント
        region = extract_region_from_arn(context.invoked_function_arn)
        host = f"bedrock-runtime.{region}.amazonaws.com"
        endpoint = f"https://{host}/model/{MODEL_ID}/invoke"

        # Lambda 実行ロールの一時認証情報
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        session_token = os.environ.get('AWS_SESSION_TOKEN')
        if not access_key or not secret_key:
            raise Exception("AWS credentials not found in environment")

        # リクエストボディの解析
        body = json.loads(event.get('body', '{}'))
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        messages = conversation_history.copy()
        messages.append({ "role": "user", "content": message })

        # Bedrock 用メッセージ形式に変換
        bedrock_messages = []
        for msg in messages:
            bedrock_messages.append({
                "role": msg["role"],
                "content": [ { "text": msg["content"] } ]
            })

        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        request_body = json.dumps(request_payload, separators=(',', ':'))

        # --- SigV4 署名の準備 ---
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        service = 'bedrock-runtime'
        method = 'POST'
        canonical_uri = f"/model/{MODEL_ID}/invoke"
        canonical_querystring = ''
        payload_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()
        canonical_headers = (
            f"content-type:application/json\n"
            f"host:{host}\n"
            f"x-amz-date:{amz_date}\n"
        )
        signed_headers = 'content-type;host;x-amz-date'
        canonical_request = '\n'.join([
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash
        ])

        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
        string_to_sign = '\n'.join([
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        ])

        signing_key = get_signature_key(secret_key, date_stamp, region, service)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        authorization_header = (
            f"{algorithm} Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # HTTP ヘッダー
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Host': host,
            'X-Amz-Date': amz_date,
            'Authorization': authorization_header,
        }
        if session_token:
            headers['X-Amz-Security-Token'] = session_token

        # API 呼び出し
        req = urllib.request.Request(
            endpoint,
            data=request_body.encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read()
        response_body = json.loads(resp_body)

        # レスポンス検証
        output = response_body.get('output', {})
        content_list = output.get('message', {}).get('content', [])
        if not content_list:
            raise Exception("No response content from the model")

        assistant_response = content_list[0].get('text', '')
        messages.append({ "role": "assistant", "content": assistant_response })

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        return {
            "statusCode": e.code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"HTTPError: {e.code} {e.reason}",
                "details": error_body
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(e)
            })
        }

