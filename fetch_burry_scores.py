"""
fetch_burry_scores.py
Burry Style Stock Screener — HerdVibe
매일 GitHub Actions에서 실행: IWM 보유 종목 → yfinance → 스코어링 → JSON 저장
"""

import json
import time
import random
import datetime
import os
import pandas as pd
import yfinance as yf
from io import StringIO

# ── 설정 ──────────────────────────────────────────────
OUTPUT_PATH = "data/burry_scores.json"
IWM_CSV_URL = "https://www.ishares.com/us/products/239710/IWM/1467271812596.ajax?tab=holdings&fileType=csv"
BATCH_SIZE = 10       # 한 번에 처리할 종목 수
SLEEP_BETWEEN = 1.5   # 배치 간 대기(초)
MAX_TICKERS = 2000    # 전체 처리 상한


# ── 1. IWM 티커 로드 ───────────────────────────────────
def load_iwm_tickers():
    """iShares IWM CSV에서 Russell 2000 구성 종목 티커 추출"""
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.ishares.com/"
        }
        r = requests.get(IWM_CSV_URL, headers=headers, timeout=30)
        r.raise_for_status()
        # 앞 9행 메타데이터 스킵
        df = pd.read_csv(StringIO(r.text), skiprows=9)
        df = df[df['Asset Class'] == 'Equity']
        df = df[df['Ticker'].notna()]
        df = df[df['Ticker'].str.strip() != '-']
        tickers = df['Ticker'].str.strip().tolist()
        sectors = dict(zip(df['Ticker'].str.strip(), df['Sector'].fillna('Unknown')))
        names   = dict(zip(df['Ticker'].str.strip(), df['Name'].fillna('')))
        print(f"[OK] IWM 구성 종목 {len(tickers)}개 로드")
        return tickers[:MAX_TICKERS], sectors, names
    except Exception as e:
        print(f"[ERROR] IWM CSV 로드 실패: {e}")
        # 폴백: 로컬 CSV
        try:
            df = pd.read_csv("data/iwm_holdings.csv", skiprows=9)
            df = df[df['Asset Class'] == 'Equity']
            df = df[df['Ticker'].notna()]
            tickers = df['Ticker'].str.strip().tolist()
            sectors = dict(zip(df['Ticker'].str.strip(), df['Sector'].fillna('Unknown')))
            names   = dict(zip(df['Ticker'].str.strip(), df['Name'].fillna('')))
            print(f"[OK] 로컬 CSV 폴백 {len(tickers)}개 로드")
            return tickers[:MAX_TICKERS], sectors, names
        except Exception as e2:
            print(f"[ERROR] 로컬 CSV도 실패: {e2}")
            return [], {}, {}


# ── 2. Hard Filter ─────────────────────────────────────
def passes_hard_filter(info):
    """
    이것 중 하나라도 해당하면 탈락 (dying / 재무위험 종목 제거)
    """
    # 시가총액 $50M 미만
    mc = info.get('marketCap', 0) or 0
    if mc < 50_000_000:
        return False, "시총 $50M 미만"

    # Debt/Equity > 3 (과도한 부채)
    de = info.get('debtToEquity')
    if de is not None and de > 300:   # yfinance는 %단위 반환 (300 = 3.0)
        return False, f"D/E {de:.0f}% 초과"

    # 영업현금흐름 음수 (최근)
    ocf = info.get('operatingCashflow')
    if ocf is not None and ocf < 0:
        return False, "영업현금흐름 음수"

    return True, "OK"


# ── 3. Burry Score 계산 ────────────────────────────────
def calc_burry_score(info, history):
    """
    100점 만점 Burry Long Score
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    P/B ≤ 1.5                    → 30점
    52주 고점 대비 -40% 이상 하락 → 20점
    Analyst 매수 비율 낮음        → 15점
    FCF 양수                      → 15점
    Debt/Equity < 50%            → 10점
    Shareholder Turnover 높음    → 10점
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    score = 0
    details = {}

    # 1) P/B Ratio (Tangible Book Value Anchor) — 30점
    pb = info.get('priceToBook')
    if pb is not None:
        if pb <= 1.0:
            s = 30
        elif pb <= 1.5:
            s = 20
        elif pb <= 2.0:
            s = 10
        else:
            s = 0
        score += s
        details['pb_ratio'] = round(pb, 2)
        details['pb_score'] = s
    else:
        details['pb_ratio'] = None
        details['pb_score'] = 0

    # 2) 52주 고점 대비 하락률 (Capitulation) — 20점
    high52 = info.get('fiftyTwoWeekHigh')
    cur_price = info.get('currentPrice') or info.get('regularMarketPrice')
    if high52 and cur_price and high52 > 0:
        drop_pct = (cur_price - high52) / high52 * 100
        if drop_pct <= -60:
            s = 20
        elif drop_pct <= -40:
            s = 15
        elif drop_pct <= -25:
            s = 8
        else:
            s = 0
        score += s
        details['drop_from_high_pct'] = round(drop_pct, 1)
        details['drop_score'] = s
    else:
        details['drop_from_high_pct'] = None
        details['drop_score'] = 0

    # 3) Analyst Consensus Negativity — 15점
    # recommendationMean: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
    rec = info.get('recommendationMean')
    if rec is not None:
        if rec >= 3.5:      # 월가가 싫어함
            s = 15
        elif rec >= 3.0:    # Hold 이하
            s = 8
        else:
            s = 0
        score += s
        details['analyst_mean'] = round(rec, 2)
        details['analyst_score'] = s
    else:
        details['analyst_mean'] = None
        details['analyst_score'] = 0

    # 4) Free Cash Flow 양수 — 15점
    fcf = info.get('freeCashflow')
    if fcf is not None:
        if fcf > 0:
            s = 15
        else:
            s = 0
        score += s
        details['fcf'] = fcf
        details['fcf_score'] = s
    else:
        details['fcf'] = None
        details['fcf_score'] = 0

    # 5) Debt/Equity 낮음 (Balance Sheet Strength) — 10점
    de = info.get('debtToEquity')
    if de is not None:
        if de < 30:         # D/E < 0.3
            s = 10
        elif de < 80:       # D/E < 0.8
            s = 6
        elif de < 150:      # D/E < 1.5
            s = 3
        else:
            s = 0
        score += s
        details['debt_to_equity'] = round(de, 1)
        details['de_score'] = s
    else:
        details['debt_to_equity'] = None
        details['de_score'] = 0

    # 6) Shareholder Turnover (3개월 거래량 / 유통주식수) — 10점
    shares = info.get('floatShares') or info.get('sharesOutstanding')
    avg_vol = info.get('averageVolume3Month') or info.get('averageVolume')
    if shares and avg_vol and shares > 0:
        # 3개월(63 거래일) 기준 turnover 배수
        turnover = (avg_vol * 63) / shares
        if turnover >= 10:
            s = 10
        elif turnover >= 5:
            s = 6
        elif turnover >= 3:
            s = 3
        else:
            s = 0
        score += s
        details['shareholder_turnover'] = round(turnover, 1)
        details['turnover_score'] = s
    else:
        details['shareholder_turnover'] = None
        details['turnover_score'] = 0

    return score, details


# ── 4. 배치 처리 ───────────────────────────────────────
def process_tickers(tickers, sectors, names):
    results = []
    failed  = []
    total = len(tickers)

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"[{i+len(batch)}/{total}] 처리 중: {batch}")

        for ticker in batch:
            try:
                tk = yf.Ticker(ticker)
                info = tk.info

                # info가 비어있으면 스킵
                if not info or info.get('quoteType') not in ('EQUITY', 'ETF', None):
                    continue
                if info.get('quoteType') == 'ETF':
                    continue

                # Hard Filter
                ok, reason = passes_hard_filter(info)
                if not ok:
                    continue

                # 최근 3개월 히스토리 (turnover 계산용)
                hist = tk.history(period="3mo")

                # Burry Score
                score, details = calc_burry_score(info, hist)

                # 결과 저장
                mc = info.get('marketCap', 0) or 0
                result = {
                    "ticker":   ticker,
                    "name":     names.get(ticker, info.get('longName', ticker)),
                    "sector":   sectors.get(ticker, info.get('sector', 'Unknown')),
                    "score":    score,
                    "price":    info.get('currentPrice') or info.get('regularMarketPrice'),
                    "market_cap": mc,
                    "details":  details,
                }
                results.append(result)

            except Exception as e:
                failed.append(ticker)
                print(f"  [SKIP] {ticker}: {e}")

        # Rate limit 방지
        time.sleep(SLEEP_BETWEEN + random.uniform(0, 0.5))

    return results, failed


# ── 5. 메인 ───────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Burry Screener 실행: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 티커 로드
    tickers, sectors, names = load_iwm_tickers()
    if not tickers:
        print("[ERROR] 티커 로드 실패. 종료.")
        return

    # 처리
    results, failed = process_tickers(tickers, sectors, names)

    # 점수 순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)

    # Burry Zone 판별 (60점 이상)
    burry_zone = [r for r in results if r['score'] >= 60]
    watchlist  = [r for r in results if 45 <= r['score'] < 60]

    # 출력 JSON
    output = {
        "updated_at":  datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "updated_kst": (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M KST"),
        "total_screened": len(results),
        "burry_zone_count": len(burry_zone),
        "watchlist_count":  len(watchlist),
        "failed_count":     len(failed),
        "all_results":  results,          # 전체 (점수순)
        "burry_zone":   burry_zone,       # 60점 이상
        "watchlist":    watchlist,        # 45~59점
        "top30":        results[:30],     # 상위 30개
    }

    # 저장
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 총 {len(results)}개 스크리닝 완료")
    print(f"  Burry Zone (60+): {len(burry_zone)}개")
    print(f"  Watchlist (45-59): {len(watchlist)}개")
    print(f"  실패: {len(failed)}개")
    print(f"  저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
