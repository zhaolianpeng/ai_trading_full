# data/market_data.py
"""
从线上交易市场获取真实数据
支持多种数据源：Yahoo Finance, Binance等
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
from utils.logger import logger
from utils.validators import validate_price_data, fix_price_data

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed. Install with: pip install ccxt")


def fetch_yahoo_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    从 Yahoo Finance 获取数据
    
    Args:
        symbol: 交易对符号（如 'BTC-USD', 'AAPL', 'TSLA'）
        period: 数据周期 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: 数据间隔 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start: 开始日期 (YYYY-MM-DD)，如果指定则覆盖 period
        end: 结束日期 (YYYY-MM-DD)
    
    Returns:
        包含 OHLCV 数据的 DataFrame，以 datetime 为索引
    
    Examples:
        # 获取比特币最近1年的小时数据
        df = fetch_yahoo_data('BTC-USD', period='1y', interval='1h')
        
        # 获取苹果股票最近3个月的数据
        df = fetch_yahoo_data('AAPL', period='3mo', interval='1d')
        
        # 获取指定日期范围的数据
        df = fetch_yahoo_data('TSLA', start='2024-01-01', end='2024-12-31', interval='1d')
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")
    
    logger.info(f"Fetching data from Yahoo Finance for {symbol}...")
    logger.info(f"Parameters: period={period}, interval={interval}, start={start}, end={end}")
    
    # 符号格式转换：BTC-USD -> BTCUSD=X (Yahoo Finance 格式)
    yahoo_symbol = symbol
    if '-' in symbol and symbol.endswith('-USD'):
        # 加密货币格式转换：BTC-USD -> BTCUSD=X
        yahoo_symbol = symbol.replace('-USD', 'USD=X')
        logger.info(f"Converting symbol format: {symbol} -> {yahoo_symbol}")
    
    try:
        ticker = yf.Ticker(yahoo_symbol)
        
        # 尝试获取数据
        if start and end:
            # 使用指定的日期范围
            df = ticker.history(start=start, end=end, interval=interval)
        elif start:
            # 只有开始日期，使用到当前
            df = ticker.history(start=start, interval=interval)
        else:
            # 使用 period
            df = ticker.history(period=period, interval=interval)
        
        # 如果数据为空，尝试备用符号格式
        if df.empty:
            logger.warning(f"No data returned for {yahoo_symbol}, trying alternative formats...")
            # 尝试原始符号
            if yahoo_symbol != symbol:
                ticker = yf.Ticker(symbol)
                if start and end:
                    df = ticker.history(start=start, end=end, interval=interval)
                elif start:
                    df = ticker.history(start=start, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
            
            # 如果仍然为空，尝试其他格式
            if df.empty and symbol.endswith('-USD'):
                # 尝试 BTCUSD=X 格式
                alt_symbol = symbol.replace('-', '') + '=X'
                logger.info(f"Trying alternative symbol: {alt_symbol}")
                ticker = yf.Ticker(alt_symbol)
                if start and end:
                    df = ticker.history(start=start, end=end, interval=interval)
                elif start:
                    df = ticker.history(start=start, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            # 对于加密货币，提供更友好的错误信息和建议
            is_crypto = any(x in symbol.upper() for x in ['BTC', 'ETH', 'USD', 'USDT', 'CRYPTO'])
            
            if is_crypto:
                error_msg = (
                    f"无法从 Yahoo Finance 获取加密货币数据 {symbol}。\n"
                    f"Yahoo Finance 对加密货币的支持可能不稳定。\n\n"
                    f"建议解决方案：\n"
                    f"  1. 使用 Binance 数据源（推荐）：\n"
                    f"     DATA_SOURCE=binance MARKET_SYMBOL=BTC/USDT MARKET_TIMEFRAME=1h\n"
                    f"  2. 使用股票数据测试（如 AAPL, TSLA）：\n"
                    f"     DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=3mo MARKET_INTERVAL=1d\n"
                    f"  3. 使用合成数据：\n"
                    f"     DATA_SOURCE=synthetic"
                )
            else:
                error_msg = (
                    f"No data returned for symbol {symbol} (tried: {yahoo_symbol}).\n"
                    f"Possible reasons:\n"
                    f"  1. Symbol format may be incorrect\n"
                    f"  2. Time period may be too long. Try shorter period (e.g., '3mo' instead of '1y')\n"
                    f"  3. Interval may not be supported. Try '1d' instead of '1h'\n"
                    f"  4. Yahoo Finance may be temporarily unavailable"
                )
            raise ValueError(error_msg)
        
        # 重命名列以匹配我们的格式
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # 只保留需要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # 确保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 数据验证和修复
        is_valid, errors = validate_price_data(df)
        if not is_valid:
            logger.warning(f"Data validation found issues: {errors}")
            logger.info("Attempting to fix data issues...")
            df = fix_price_data(df)
        
        logger.info(f"Fetched {len(df)} rows from Yahoo Finance")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {e}")
        raise


def fetch_binance_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    从 Binance 获取加密货币数据
    
    Args:
        symbol: 交易对符号（如 'BTC/USDT', 'ETH/USDT'）
        timeframe: 时间框架 ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
        limit: 最大数据条数（默认1000，最大1000）
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        包含 OHLCV 数据的 DataFrame，以 datetime 为索引
    
    Examples:
        # 获取比特币最近1000条小时数据
        df = fetch_binance_data('BTC/USDT', timeframe='1h', limit=1000)
    """
    if not CCXT_AVAILABLE:
        raise ImportError("ccxt is not installed. Install with: pip install ccxt")
    
    logger.info(f"Fetching data from Binance for {symbol}...")
    
    try:
        # 判断是否为永续合约（通过符号判断，BTC/USDT等通常是永续）
        # 用户可以通过环境变量 MARKET_TYPE 指定：'spot', 'future', 'swap'
        market_type = os.getenv('MARKET_TYPE', 'future' if 'USDT' in symbol else 'spot')
        
        # 对于BTC/USDT等，默认使用永续合约
        if 'USDT' in symbol and market_type == 'spot':
            logger.warning(f"检测到 {symbol} 但 MARKET_TYPE=spot，建议使用 MARKET_TYPE=future 获取永续合约价格")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': market_type,  # 'spot', 'future', 'swap'
            }
        })
        
        logger.info(f"使用 Binance {market_type} 市场获取 {symbol} 数据")
        
        # 转换时间框架格式
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # 获取数据（优先获取最新数据）
        # 注意：不指定since参数，直接获取最新的limit条数据，确保获取到最新价格
        try:
            # 方法1：直接获取最新数据（推荐）
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 如果指定了start_time，且获取的数据最早时间早于start_time，则过滤
            if start_time and ohlcv:
                start_timestamp = int(start_time.timestamp() * 1000)
                # 过滤掉早于start_time的数据
                ohlcv = [candle for candle in ohlcv if candle[0] >= start_timestamp]
                
                # 如果过滤后数据太少，尝试从start_time开始获取
                if len(ohlcv) < limit // 2:
                    logger.info(f"过滤后数据较少，从 {start_time} 开始获取数据...")
                    since = int(start_time.timestamp() * 1000)
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            logger.warning(f"获取最新数据失败，尝试从指定时间开始获取: {e}")
            if start_time:
                since = int(start_time.timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            raise ValueError(f"No data returned for symbol {symbol}")
        
        # 转换为 DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # 数据验证和修复
        is_valid, errors = validate_price_data(df)
        if not is_valid:
            logger.warning(f"Data validation found issues: {errors}")
            logger.info("Attempting to fix data issues...")
            df = fix_price_data(df)
        
        logger.info(f"Fetched {len(df)} rows from Binance")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # 获取最新的ticker价格，确保数据是最新的
        last_kline_time = df.index[-1]
        last_kline_price = df['close'].iloc[-1]
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            latest_price = ticker.get('last', None)
            ticker_time = ticker.get('timestamp', None)
            
            if latest_price:
                # 格式化ticker时间
                if ticker_time:
                    ticker_dt = pd.to_datetime(ticker_time, unit='ms')
                    logger.info(f"✅ 最新ticker价格: ${latest_price:.2f} (时间: {ticker_dt})")
                else:
                    logger.info(f"✅ 最新ticker价格: ${latest_price:.2f}")
                
                # 如果最新ticker价格与最后一条K线价格差异较大，记录警告
                price_diff_pct = abs(latest_price - last_kline_price) / latest_price * 100
                
                if price_diff_pct > 1.0:  # 如果差异超过1%
                    logger.error(f"❌ 严重价格差异: ticker价格 ${latest_price:.2f} vs K线价格 ${last_kline_price:.2f} (差异 {price_diff_pct:.2f}%)")
                    logger.error(f"   可能原因:")
                    logger.error(f"   1. 使用了错误的市场类型 (当前: {market_type})")
                    logger.error(f"   2. K线数据不是最新的")
                    logger.error(f"   3. 符号格式不正确 (当前: {symbol})")
                    logger.error(f"   建议: 检查 MARKET_TYPE 环境变量，确保设置为 'future'")
                elif price_diff_pct > 0.1:  # 如果差异超过0.1%
                    logger.warning(f"⚠️ 价格差异: ticker价格 ${latest_price:.2f} vs K线价格 ${last_kline_price:.2f} (差异 {price_diff_pct:.2f}%)")
                    logger.warning(f"   这可能是因为K线数据不是最新的，或者使用了不同的市场类型")
        except Exception as e:
            logger.warning(f"无法获取最新ticker价格: {e}")
        
        logger.info(f"📊 最后K线价格: ${last_kline_price:.2f} (时间: {last_kline_time})")
        
        # 验证价格是否在合理范围内（BTC价格应该在10000-200000之间）
        if last_kline_price < 10000 or last_kline_price > 200000:
            logger.error(f"❌ K线价格异常: ${last_kline_price:.2f}")
            logger.error(f"   BTC价格应该在 $10,000 - $200,000 之间")
            logger.error(f"   当前市场类型: {market_type}")
            logger.error(f"   当前符号: {symbol}")
            logger.error(f"   建议检查 MARKET_TYPE 环境变量")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Binance: {e}")
        raise


def fetch_market_data(
    symbol: str,
    data_source: str = "yahoo",
    **kwargs
) -> pd.DataFrame:
    """
    统一的市场数据获取接口
    
    Args:
        symbol: 交易对符号
        data_source: 数据源 ('yahoo', 'binance')
        **kwargs: 传递给具体数据源的参数
    
    Returns:
        包含 OHLCV 数据的 DataFrame
    
    Examples:
        # 从 Yahoo Finance 获取数据
        df = fetch_market_data('BTC-USD', data_source='yahoo', period='1y', interval='1h')
        
        # 从 Binance 获取数据
        df = fetch_market_data('BTC/USDT', data_source='binance', timeframe='1h', limit=1000)
    """
    if data_source.lower() == 'yahoo':
        return fetch_yahoo_data(symbol, **kwargs)
    elif data_source.lower() == 'binance':
        return fetch_binance_data(symbol, **kwargs)
    else:
        raise ValueError(f"Unsupported data source: {data_source}. Supported: 'yahoo', 'binance'")


def get_popular_symbols() -> dict:
    """
    返回常用的交易对符号列表
    
    Returns:
        包含不同类别交易对的字典
    """
    return {
        'crypto': {
            'BTC-USD': 'Bitcoin (Yahoo Finance)',
            'ETH-USD': 'Ethereum (Yahoo Finance)',
            'BTC/USDT': 'Bitcoin (Binance)',
            'ETH/USDT': 'Ethereum (Binance)',
        },
        'stocks': {
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'NVDA': 'NVIDIA',
        },
        'indices': {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
        }
    }

