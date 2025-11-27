# 市场结构判断规则

## 一、市场结构标签判断

基于K线数据，使用确定性规则判断市场结构：

**判断规则**：

1. **TREND_UP（上升趋势）**：
   - `close[t] > EMA21[t] > EMA55[t]`
   - `(close[t] - close[t-20]) / close[t-20] > 0.02`（20根K线上涨超过2%）
   - `trend_strength > 60`

2. **TREND_DOWN（下降趋势）**：
   - `close[t] < EMA21[t] < EMA55[t]`
   - `(close[t] - close[t-20]) / close[t-20] < -0.02`（20根K线下跌超过2%）
   - `trend_strength > 60`

3. **RANGE（震荡）**：
   - `trend_strength < 40`
   - `(high_20[t] - low_20[t]) / close[t] < 0.03`（20根K线波动范围小于3%）
   - EMA排列不明确

4. **REVERSAL_UP（反转上涨）**：
   - `close[t] > close[t-1] > close[t-2]`（连续上涨）
   - `close[t-10] < close[t-20]`（之前下跌）
   - `RSI14[t] < 50 且 RSI14[t] > RSI14[t-3]`（RSI从低位反弹）

5. **REVERSAL_DOWN（反转下跌）**：
   - `close[t] < close[t-1] < close[t-2]`（连续下跌）
   - `close[t-10] > close[t-20]`（之前上涨）
   - `RSI14[t] > 50 且 RSI14[t] < RSI14[t-3]`（RSI从高位回落）

6. **BREAKOUT_UP（向上突破）**：
   - `close[t] > high_20[t] * 0.995`（突破20根K线高点）
   - `volume[t] / vol_ma50[t] >= 1.2`（成交量放大）
   - `close[t-1] <= high_20[t]`（前一根未突破）

7. **BREAKOUT_DOWN（向下突破）**：
   - `close[t] < low_20[t] * 1.005`（跌破20根K线低点）
   - `volume[t] / vol_ma50[t] >= 1.2`（成交量放大）
   - `close[t-1] >= low_20[t]`（前一根未跌破）

## 二、多时间周期市场结构一致性判断

**输出格式**：`{1m: ___, 5m: ___, 15m: ___, 1h: ___, 4h: ___, 一致性评分:0-100}`

**一致性评分计算**：
```
一致性评分 = (相同标签数量 / 总周期数) * 100

例如：
- 如果5个周期都是TREND_UP，一致性评分 = 100
- 如果4个周期是TREND_UP，1个是RANGE，一致性评分 = 80
- 如果3个周期是TREND_UP，2个是RANGE，一致性评分 = 60
```

## 三、波动率Regime判断

基于过去N根K线的实体、上下影、ATR、波动聚集情况：

**判断规则**：

1. **LOW_VOL（低波动）**：
   - `(ATR14[t] / close[t]) * 100 < 0.5%`
   - `(high_20[t] - low_20[t]) / close[t] < 1%`
   - `ATR14[t] < ATR14_MA20[t] * 0.7`

2. **MID_VOL（中等波动）**：
   - `0.5% <= (ATR14[t] / close[t]) * 100 <= 3%`
   - `1% <= (high_20[t] - low_20[t]) / close[t] <= 5%`
   - `0.7 <= ATR14[t] / ATR14_MA20[t] <= 1.5`

3. **HIGH_VOL（高波动）**：
   - `3% < (ATR14[t] / close[t]) * 100 <= 5%`
   - `5% < (high_20[t] - low_20[t]) / close[t] <= 10%`
   - `1.5 < ATR14[t] / ATR14_MA20[t] <= 2.5`

4. **EXTREME_VOL（极端波动）**：
   - `(ATR14[t] / close[t]) * 100 > 5%`
   - `(high_20[t] - low_20[t]) / close[t] > 10%`
   - `ATR14[t] / ATR14_MA20[t] > 2.5`

## 四、反转结构判断（最近20根K线）

**判断规则**：

1. **REVERSAL_UP（反转上涨）**：
   - `close[t] > close[t-1] > close[t-2]`（最近3根上涨）
   - `close[t-10] < close[t-20]`（10根K线前比20根K线前低）
   - `close[t] > close[t-10]`（当前价格高于10根K线前）
   - `RSI14[t] < 50 且 RSI14[t] > RSI14[t-5]`（RSI从低位反弹）

2. **REVERSAL_DOWN（反转下跌）**：
   - `close[t] < close[t-1] < close[t-2]`（最近3根下跌）
   - `close[t-10] > close[t-20]`（10根K线前比20根K线前高）
   - `close[t] < close[t-10]`（当前价格低于10根K线前）
   - `RSI14[t] > 50 且 RSI14[t] < RSI14[t-5]`（RSI从高位回落）

3. **NONE（无反转）**：
   - 不满足上述条件

## 五、突破有效性判断

**判断规则**：

1. **VALID_BREAKOUT（有效突破）**：
   - 突破后连续3根K线收盘价都在突破位上方（向上突破）或下方（向下突破）
   - `volume[t] / vol_ma50[t] >= 1.2`（突破时成交量放大）
   - `ATR14[t] / ATR14_MA20[t] >= 0.8`（波动率不低于历史均值）
   - 突破K线实体占比 >= 0.5

2. **FAKE_BREAKOUT（假突破）**：
   - 突破后3根K线内价格回落到突破位另一侧
   - 或 `volume[t] / vol_ma50[t] < 1.2`（突破时成交量未放大）
   - 或 `ATR14[t] / ATR14_MA20[t] < 0.8`（波动率低于历史均值）
   - 或 突破K线实体占比 < 0.5

## 六、趋势健康度评分（0~100）

基于均线角度、一致性、K线形态、波动率：

**计算公式**：
```
趋势健康度 = 均线角度得分 + 一致性得分 + K线形态得分 + 波动率得分

均线角度得分（0-30分）：
  - 如果 close > EMA21 > EMA55 > EMA100: 30分
  - 如果 close > EMA21 > EMA55: 20分
  - 如果 close > EMA21: 10分
  - 否则: 0分

一致性得分（0-30分）：
  - 过去20根K线中上涨K线占比 * 30

K线形态得分（0-20分）：
  - 如果 higher_highs 且 higher_lows: 20分
  - 如果 higher_highs 或 higher_lows: 10分
  - 否则: 0分

波动率得分（0-20分）：
  - 如果 1% <= (ATR14 / close) * 100 <= 3%: 20分
  - 如果 0.5% <= (ATR14 / close) * 100 < 1% 或 3% < (ATR14 / close) * 100 <= 5%: 15分
  - 否则: 5分
```

## 七、结构切换判断（trend→range 或 range→trend）

**判断规则**：

将数据分为两段：前半段（前50根K线）和后半段（后50根K线）

1. **前半段市场类型**：
   - 如果 `trend_strength(前半段) > 50`: TREND
   - 否则: RANGE

2. **后半段市场类型**：
   - 如果 `trend_strength(后半段) > 50`: TREND
   - 否则: RANGE

3. **结构切换判断**：
   - 如果前半段和后半段的市场类型不同: YES
   - 否则: NO

