import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
import jquantsapi

load_dotenv()


def get_client() -> jquantsapi.ClientV2:
    api_key = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not api_key:
        print("エラー: .env に JQUANTS_REFRESH_TOKEN が設定されていません。")
        sys.exit(1)
    return jquantsapi.ClientV2(api_key=api_key)


def fetch_prime_stocks() -> None:
    print("J-Quants API (V2) に接続中...")
    try:
        client = get_client()
        df = client.get_eq_master()
    except Exception as e:
        print(f"エラー: API接続に失敗しました。\n詳細: {e}")
        sys.exit(1)

    prime = df[df["MktNm"] == "プライム"].reset_index(drop=True)

    if prime.empty:
        print("警告: 東証プライム銘柄が見つかりませんでした。")
        return

    print(f"\n東証プライム銘柄数: {len(prime)} 件")
    print("\n── 先頭5件 ──")
    print(prime[["Code", "CoName", "MktNm", "S17Nm"]].head().to_string(index=False))


if __name__ == "__main__":
    fetch_prime_stocks()
