#!/usr/bin/env python
"""
BTC 분석기 - 사용자 관리 도구
실행: python setup_users.py
"""

import bcrypt
import yaml
import getpass
import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.yaml"


# ── 유틸 함수 ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """config.yaml 로드 (없으면 기본 구조 생성)"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "credentials": {"usernames": {}},
        "cookie": {
            "expiry_days": 30,
            "key": "change-this-key-in-production",
            "name": "btc_analyzer_auth",
        },
    }


def save_config(cfg: dict) -> None:
    """config.yaml 저장"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"\n✅ {CONFIG_FILE.name} 저장 완료")


def hash_password(password: str) -> str:
    """bcrypt 해시 생성"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def check_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def divider():
    print("─" * 50)


# ── 기능 함수 ──────────────────────────────────────────────────────────────────

def list_users(cfg: dict) -> None:
    """사용자 목록 출력"""
    users = cfg["credentials"]["usernames"]
    print()
    divider()
    if not users:
        print("  (등록된 사용자 없음)")
    else:
        print(f"  {'아이디':<16} {'이름':<18} {'이메일'}")
        divider()
        for uid, info in users.items():
            print(f"  {uid:<16} {info.get('name',''):<18} {info.get('email', '-')}")
    divider()
    print()


def add_user(cfg: dict) -> None:
    """새 사용자 추가"""
    print("\n=== 새 사용자 추가 ===")
    username = input("아이디 (영문·숫자): ").strip()
    if not username:
        print("❌ 아이디를 입력해주세요.")
        return
    if username in cfg["credentials"]["usernames"]:
        print(f"❌ 이미 존재하는 아이디: {username}")
        return

    name = input("이름: ").strip()
    email = input("이메일 (선택): ").strip()

    password = getpass.getpass("비밀번호 (최소 8자): ")
    if len(password) < 8:
        print("❌ 비밀번호는 최소 8자 이상이어야 합니다.")
        return
    password2 = getpass.getpass("비밀번호 확인: ")
    if password != password2:
        print("❌ 비밀번호가 일치하지 않습니다.")
        return

    print("  해시 생성 중...", end="", flush=True)
    cfg["credentials"]["usernames"][username] = {
        "name": name,
        "email": email,
        "password": hash_password(password),
    }
    print(" 완료")
    save_config(cfg)
    print(f"✅ 사용자 '{username}' ({name}) 추가 완료!")


def change_password(cfg: dict) -> None:
    """비밀번호 변경"""
    print("\n=== 비밀번호 변경 ===")
    username = input("아이디: ").strip()
    if username not in cfg["credentials"]["usernames"]:
        print(f"❌ 존재하지 않는 아이디: {username}")
        return

    password = getpass.getpass("새 비밀번호 (최소 8자): ")
    if len(password) < 8:
        print("❌ 비밀번호는 최소 8자 이상이어야 합니다.")
        return
    password2 = getpass.getpass("새 비밀번호 확인: ")
    if password != password2:
        print("❌ 비밀번호가 일치하지 않습니다.")
        return

    print("  해시 생성 중...", end="", flush=True)
    cfg["credentials"]["usernames"][username]["password"] = hash_password(password)
    print(" 완료")
    save_config(cfg)
    print(f"✅ '{username}' 비밀번호 변경 완료!")


def remove_user(cfg: dict) -> None:
    """사용자 삭제"""
    print("\n=== 사용자 삭제 ===")
    list_users(cfg)
    username = input("삭제할 아이디: ").strip()
    if username not in cfg["credentials"]["usernames"]:
        print(f"❌ 존재하지 않는 아이디: {username}")
        return
    info = cfg["credentials"]["usernames"][username]
    confirm = input(
        f"  '{username}' ({info.get('name','')})을 삭제하시겠습니까? (yes 입력): "
    ).strip()
    if confirm.lower() != "yes":
        print("취소되었습니다.")
        return
    del cfg["credentials"]["usernames"][username]
    save_config(cfg)
    print(f"✅ 사용자 '{username}' 삭제 완료!")


def verify_password_cmd(cfg: dict) -> None:
    """비밀번호 검증 (테스트용)"""
    print("\n=== 비밀번호 검증 ===")
    username = input("아이디: ").strip()
    if username not in cfg["credentials"]["usernames"]:
        print(f"❌ 존재하지 않는 아이디: {username}")
        return
    stored_hash = cfg["credentials"]["usernames"][username]["password"]
    password = getpass.getpass("비밀번호: ")
    if check_password(password, stored_hash):
        print("✅ 비밀번호 일치!")
    else:
        print("❌ 비밀번호 불일치")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("══════════════════════════════════════")
    print("   ₿ BTC 분석기 · 사용자 관리 도구")
    print("══════════════════════════════════════")
    print(f"   설정 파일: {CONFIG_FILE}")

    cfg = load_config()

    while True:
        print("\n  1. 사용자 목록 보기")
        print("  2. 새 사용자 추가")
        print("  3. 비밀번호 변경")
        print("  4. 사용자 삭제")
        print("  5. 비밀번호 검증 (테스트)")
        print("  0. 종료")
        print()
        choice = input("선택: ").strip()

        if choice == "1":
            list_users(cfg)
        elif choice == "2":
            add_user(cfg)
            cfg = load_config()
        elif choice == "3":
            change_password(cfg)
            cfg = load_config()
        elif choice == "4":
            remove_user(cfg)
            cfg = load_config()
        elif choice == "5":
            verify_password_cmd(cfg)
        elif choice == "0":
            print("\n종료합니다. 👋\n")
            sys.exit(0)
        else:
            print("  잘못된 선택입니다.")


if __name__ == "__main__":
    main()
