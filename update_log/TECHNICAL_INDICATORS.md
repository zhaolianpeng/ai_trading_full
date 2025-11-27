# 技术指标实现清单

本文档列出项目中所有已实现的技术指标及其代码位置。

## 📊 基础技术指标 (`features/ta_basic.py`)

### 1. EMA (指数移动平均线)
- **函数**: `ema(series, n)`
- **实现**: `series.ewm(span=n, adjust=False).mean()`
- **已添加的指标**:
  - `ema21` - 21周期EMA
  - `ema55` - 55周期EMA
  - `ema100` - 100周期EMA
  - `ema200` - 200周期EMA
  - `ema144` - 144周期EMA (维加斯通道)
  - `ema169` - 169周期EMA (维加斯通道)

### 2. RSI (相对强弱指标)
- **函数**: `rsi(series, n=14)`
- **实现**: 基于价格变化的指数移动平均
- **已添加的指标**:
  - `rsi14` - 14周期RSI

### 3. ATR (平均真实波幅)
- **函数**: `atr(df, n=14)`
- **实现**: 基于最高价、最低价、收盘价计算真实波幅
- **已添加的指标**:
  - `atr14` - 14周期ATR

### 4. 成交量指标
- **函数**: `df['volume'].rolling(50, min_periods=1).mean()`
- **已添加的指标**:
  - `vol_ma50` - 50周期成交量移动平均

### 5. 阻力位/支撑位
- **函数**: `df['close'].rolling(50).max()`
- **已添加的指标**:
  - `res50` - 50周期最高价（阻力位）

## 📈 高级技术指标 (`features/ta_advanced.py`)

### 1. MACD (移动平均收敛散度)
- **函数**: `macd(series, fast=12, slow=26, signal=9)`
- **已添加的指标**:
  - `macd` - MACD线
  - `macd_signal` - 信号线
  - `macd_hist` - 柱状图

### 2. 布林带 (Bollinger Bands)
- **函数**: `bollinger_bands(series, n=20, num_std=2)`
- **已添加的指标**:
  - `bb_upper` - 上轨
  - `bb_middle` - 中轨
  - `bb_lower` - 下轨
  - `bb_width` - 布林带宽度

### 3. 随机指标 (Stochastic Oscillator)
- **函数**: `stochastic(high, low, close, k_period=14, d_period=3)`
- **已添加的指标**:
  - `stoch_k` - %K值
  - `stoch_d` - %D值

### 4. 威廉指标 (Williams %R)
- **函数**: `williams_r(high, low, close, period=14)`
- **已添加的指标**:
  - `williams_r` - Williams %R值

### 5. CCI (商品通道指标)
- **函数**: `cci(high, low, close, period=20)`
- **已添加的指标**:
  - `cci` - CCI值

### 6. ADX (平均趋向指标)
- **函数**: `adx(high, low, close, period=14)`
- **已添加的指标**:
  - `adx` - ADX值
  - `plus_di` - +DI值
  - `minus_di` - -DI值

## 🎯 Eric 策略指标 (`features/eric_indicators.py`)

### 1. Eric Score
- **文件**: `features/eric_score.py`
- **功能**: 超买超卖指标，基于价格在区间内的位置
- **已添加的指标**:
  - `eric_score` - 原始Eric Score
  - `eric_score_smoothed` - 平滑后的Eric Score

### 2. Donchian 通道
- **文件**: `features/donchian_channel.py`
- **功能**: 55周期通道，用于趋势过滤
- **已添加的指标**:
  - `donchian_upper` - 上轨
  - `donchian_lower` - 下轨
  - `donchian_trend` - 通道趋势（上升/下降/横盘）

### 3. EMA 眼
- **文件**: `features/ema_eye.py`
- **功能**: 价格与EMA的相对距离，判断支撑/压力
- **已添加的指标**:
  - `ema_eye` - EMA眼值（价格与EMA的距离百分比）

### 4. 量能爆发
- **文件**: `features/volume_spike.py`
- **功能**: 成交量分析，检测爆量
- **已添加的指标**:
  - `volume_spike_level` - 爆量级别（一级/二级）

### 5. 背离检测
- **文件**: `features/eric_divergence.py`
- **功能**: 价格与Eric Score的背离
- **已添加的指标**:
  - `bullish_divergence` - 牛背离
  - `bearish_divergence` - 空背离

### 6. 波动预警
- **文件**: `features/volatility_warning.py`
- **功能**: ATR分析，高波动预警
- **已添加的指标**:
  - `volatility_warning` - 波动预警级别

## 🔍 其他特征

### 1. RSI 背离检测
- **文件**: `features/divergence.py`
- **功能**: 价格与RSI的背离检测

### 2. 市场结构分析
- **文件**: `strategy/market_structure_analyzer.py`
- **功能**: 市场结构、趋势强度、市场情绪等分析
- **函数**:
  - `calculate_trend_strength()` - 趋势强度评分（0-100）
  - `classify_market_regime()` - 市场类型分类
  - `analyze_market_structure()` - 市场结构分析
  - `analyze_market_sentiment()` - 市场情绪分析
  - `calculate_reversal_probability()` - 反转概率
  - `detect_structure_switch()` - 结构切换检测
  - `generate_quantitative_features()` - 生成10个量化特征

## 📋 指标使用方式

### 在代码中添加指标

```python
from features.ta_basic import add_basic_ta
from features.ta_advanced import add_advanced_ta
from features.eric_indicators import add_eric_indicators

# 添加基础指标
df = add_basic_ta(df)

# 添加高级指标（可选）
df = add_advanced_ta(df)

# 添加Eric指标（可选）
df = add_eric_indicators(df, 
    use_eric_score=True,
    use_donchian=True,
    use_ema_eye=True,
    use_volume_spike=True,
    use_divergence=True,
    use_volatility_warning=True
)
```

## 📊 指标分类总结

### 趋势类指标
- EMA (21, 55, 100, 200, 144, 169)
- MACD
- ADX (+DI, -DI)
- 趋势强度评分

### 动量类指标
- RSI
- 随机指标 (Stochastic)
- 威廉指标 (Williams %R)
- CCI
- Eric Score

### 波动率类指标
- ATR
- 布林带
- 波动预警

### 成交量类指标
- 成交量移动平均 (vol_ma50)
- 量能爆发

### 结构类指标
- Donchian 通道
- EMA 眼
- 阻力位/支撑位 (res50)

### 背离类指标
- RSI 背离
- Eric Score 背离

## 🔧 指标计算特点

1. **无未来函数**: 所有指标仅使用当前及历史数据
2. **向量化计算**: 使用 pandas 向量化操作，性能高效
3. **可配置**: 大部分指标支持自定义周期参数
4. **标准化**: 所有指标都有明确的数学定义

## 📝 注意事项

1. **数据要求**: 确保数据包含 `open`, `high`, `low`, `close`, `volume` 列
2. **数据量**: 某些指标需要足够的历史数据（如 EMA200 需要至少200根K线）
3. **NaN处理**: 指标计算初期可能出现NaN值，这是正常的
4. **性能**: 指标计算已优化，但大量指标可能影响性能

