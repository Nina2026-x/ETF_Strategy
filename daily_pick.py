# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import time
import requests
import json
import math
import os

# ==================== 代理配置 ====================


# ==================== 配置 ====================
# 从环境变量读取 Webhook（GitHub Secrets 传入，本地运行时需手动设置）
WEBHOOK_URL = os.environ.get('WECHAT_WEBHOOK', '')

# 如果环境变量没有设置，可以在这里临时填写（仅本地测试用）
# 正式部署到 GitHub 时，务必删除下面这行，或注释掉
if not WEBHOOK_URL:
    WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的测试key"

# 完整的 ETF 池（33只，含银华日利）
ETF_POOL = {
    '511880': '银华日利',
    '518880': '黄金ETF',
    '161226': '国投白银LOF',
    '159980': '有色ETF',
    '501018': '南方原油',
    '159985': '豆粕ETF',
    '513100': '纳指ETF',
    '513400': '道琼斯ETF',
    '159509': '纳指科技ETF',
    '159518': '标普油气ETF',
    '159529': '标普消费ETF',
    '513290': '纳指生物ETF',
    '520830': '沙特ETF',
    '513520': '日经ETF',
    '513030': '德国ETF',
    '513090': '香港证券ETF',
    '513180': '恒生科技ETF',
    '513120': '港股创新药ETF',
    '513190': '港股金融ETF',
    '159502': '标普生物科技ETF',
    '510900': '恒生国企ETF',
    '513630': '香港红利ETF',
    '159323': '港股汽车ETF',
    '513970': '恒生消费ETF',
    '515880': '通信ETF',
    '517520': '黄金股ETF',
    '515220': '煤炭ETF',
    '515050': '5GETF',
    '561330': '矿业ETF',
    '159981': '能源化工ETF',
    '510500': '中证500ETF',
    '510300': '沪深300ETF',
    '511380': '可转债ETF',
    '159915': '创业板ETF',
}

# 策略参数
LOOKBACK_DAYS = 25
MA_FILTER_DAYS = 20
SHORT_LOOKBACK_DAYS = 10
RSI_PERIOD = 6
RSI_THRESHOLD = 98
LOSS_THRESHOLD = 0.97

# ==================== 辅助函数 ====================
def send_wechat(text):
    """发送文本消息到企业微信"""
    if not WEBHOOK_URL:
        print("未设置 Webhook 地址，无法发送消息")
        return
    try:
        data = {"msgtype": "text", "text": {"content": text}}
        requests.post(WEBHOOK_URL, json=data, timeout=5)
        print("消息发送成功")
    except Exception as e:
        print("发送失败:", e)

def get_realtime_prices():
    """获取所有ETF的实时行情（最新价）"""
    try:
        df = ak.fund_etf_spot_em()
        prices = {}
        for _, row in df.iterrows():
            code = row['代码']
            if code in ETF_POOL:
                price = row['最新价']
                if price and price > 0:
                    prices[code] = price
        print(f"获取到 {len(prices)} 只ETF实时价格")
        return prices
    except Exception as e:
        print("获取实时行情失败:", e)
        return {}

def get_historical_data(code, days=LOOKBACK_DAYS+MA_FILTER_DAYS+30):
    """获取历史收盘价（前复权）"""
    try:
        end_date = datetime.date.today().strftime('%Y%m%d')
        start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y%m%d')
        df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                 start_date=start_date, end_date=end_date,
                                 adjust='qfq')
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df['日期'])
        return df['收盘']
    except Exception as e:
        return None

def calculate_rsi(prices, period=6):
    """计算RSI"""
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_score(code, realtime_price, hist_prices):
    """计算单只ETF的动量得分，返回得分和是否通过"""
    # 货币ETF特殊处理
    if code == '511880':
        return 0.001, True

    if hist_prices is None or len(hist_prices) < LOOKBACK_DAYS + MA_FILTER_DAYS:
        return None, False

    # 构建价格序列
    price_series = hist_prices.iloc[-(LOOKBACK_DAYS + MA_FILTER_DAYS):].values
    price_series = np.append(price_series, realtime_price)

    # 1. 均线过滤
    ma20 = np.mean(price_series[-MA_FILTER_DAYS:])
    if realtime_price < ma20:
        return None, False

    # 2. 短期动量过滤
    if len(price_series) >= SHORT_LOOKBACK_DAYS + 1:
        start_price = price_series[-(SHORT_LOOKBACK_DAYS + 1)]
        short_ret = realtime_price / start_price - 1
        if short_ret < 0:
            return None, False

    # 3. RSI过滤
    rsi = calculate_rsi(price_series, RSI_PERIOD)
    if rsi > RSI_THRESHOLD:
        ma5 = np.mean(price_series[-5:])
        if realtime_price < ma5:
            return None, False

    # 4. 波动风控
    if len(price_series) >= 4:
        day1 = price_series[-1] / price_series[-2]
        day2 = price_series[-2] / price_series[-3]
        day3 = price_series[-3] / price_series[-4]
        min_ratio = min(day1, day2, day3)
        if min_ratio < LOSS_THRESHOLD:
            return None, False

    # 5. 加权动量得分
    recent = price_series[-(LOOKBACK_DAYS + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    annualized = math.exp(slope * 250) - 1
    score = annualized * r_squared

    if score < 0 or score > 5:
        return None, False
    return score, True

# ==================== 主流程 ====================
def main():
    print(f"{datetime.datetime.now()} 开始尾盘选股...")

    # 1. 获取实时价格
    real_prices = get_realtime_prices()
    if not real_prices:
        send_wechat("获取实时行情失败，请检查网络")
        return

    # 2. 遍历所有ETF，计算得分
    scores = {}
    for code, name in ETF_POOL.items():
        if code not in real_prices:
            continue
        hist = get_historical_data(code)
        if hist is None:
            continue
        score, passed = calculate_score(code, real_prices[code], hist)
        if passed and score is not None:
            scores[code] = (score, name, real_prices[code])

    # 3. 输出结果
    if not scores:
        msg = "今日无符合条件的ETF，建议持有银华日利或空仓"
        send_wechat(msg)
        print(msg)
        return

    # 4. 找出得分最高的
    best_code = max(scores, key=lambda x: scores[x][0])
    best_name = scores[best_code][1]
    best_score = scores[best_code][0]
    best_price = scores[best_code][2]

    # 5. 发送微信消息
    msg = f"""【尾盘选股结果】
时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
建议持有：{best_code} {best_name}
动量得分：{best_score:.4f}
当前价格：{best_price:.3f}
"""
    send_wechat(msg)
    print(msg)

if __name__ == "__main__":
    main()
