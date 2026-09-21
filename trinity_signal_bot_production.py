#!/usr/bin/env python3
"""
TRINITY Signal Bot v3.0 - PRODUCTION (AlphaVantage Data)
========================================================

NQ futures signal bot for Discord.
Uses AlphaVantage for real-time market data.

TRINITY:
    1. SWEEP
    2. CONFIRMATION
    3. ENTRY

Designed for 24/7 scanning and manual Discord triggering.
Deployed on Railway for continuous operation.

IMPORTANT:
- This bot generates technical signals. It does not execute trades.
- Backtest results are not guarantees of future performance.
- Market data is sourced from AlphaVantage (real-time data).
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List

import discord
from discord.ext import commands, tasks
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv
import requests
import time

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

SYMBOL = os.getenv("SYMBOL", "NQ=F")

POST_HOUR = int(os.getenv("POST_HOUR", "8"))
POST_MINUTE = int(os.getenv("POST_MINUTE", "0"))

MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "7"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))

TIMEZONE = pytz.timezone("US/Eastern")

# NQ futures contract specifications
NQ_DOLLARS_PER_POINT = 20.0

# TRINITY parameters
ATR_PERIOD = 14
SWING_LOOKBACK = 20

SWEEP_ATR_MULTIPLIER = 0.20
DISPLACEMENT_ATR_MULTIPLIER = 1.50

OTE_MIN = 0.50
OTE_MAX = 0.705

STOP_ATR_MULTIPLIER = 1.0

MAX_STOP_LOSS_POINTS = 25.0

TP1_R = 1.0
TP2_R = 1.5
TP3_R = 2.0

SETUP_SCAN_BARS = 100
STRUCTURE_LOOKBACK = 20
MAX_SETUP_AGE_BARS = 12

# AlphaVantage API
ALPHAVANTAGE_API_BASE = "https://www.alphavantage.co/query"
ALPHAVANTAGE_TIMEOUT = 30

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("trinity")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Sweep:
    direction: str
    sweep_level: float
    sweep_extreme: float
    candle_index: int
    atr: float


@dataclass
class Confirmation:
    direction: str
    confirmation_type: str
    candle_index: int
    candle_high: float
    candle_low: float
    candle_open: float
    candle_close: float
    candle_range: float
    atr_multiple: float


@dataclass
class Signal:
    valid: bool
    direction: Optional[str]
    confidence: int
    current_price: float

    sweep: Optional[Sweep]
    confirmation: Optional[Confirmation]

    impulse_high: Optional[float]
    impulse_low: Optional[float]

    entry_low: Optional[float]
    entry_high: Optional[float]

    stop_loss: Optional[float]

    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]

    risk_points: Optional[float]
    risk_dollars_per_contract: Optional[float]

    generated_at: datetime

    fvg_info: Optional[Dict[str, Any]] = None

    reason: Optional[str] = None


# ============================================================
# ALPHAVANTAGE DATA FETCHER
# ============================================================

class AlphaVantageDataFetcher:
    """Fetches real-time market data from AlphaVantage"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = ALPHAVANTAGE_API_BASE
        self.tz = TIMEZONE

    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "1min",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data from AlphaVantage
        
        symbol: e.g., "NQ=F", "AAPL", "EUR/USD"
        interval: "1min", "5min", "15min", "30min", "60min"
        limit: number of recent candles to fetch
        """

        try:
            logger.info(
                "Fetching %s (%s) from AlphaVantage...",
                symbol,
                interval
            )

            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "full"
            }

            response = requests.get(
                self.base_url,
                params=params,
                timeout=ALPHAVANTAGE_TIMEOUT
            )

            if response.status_code != 200:
                logger.error(
                    "AlphaVantage API error %d: %s",
                    response.status_code,
                    response.text
                )
                return None

            data = response.json()

            # Check for errors in response
            if "Error Message" in data:
                logger.error("AlphaVantage error: %s", data["Error Message"])
                return None

            if "Note" in data:
                logger.warning("AlphaVantage rate limit: %s", data["Note"])
                time.sleep(1)
                return None

            # Find the time series key
            ts_key = None
            for key in data.keys():
                if key.startswith("Time Series"):
                    ts_key = key
                    break

            if not ts_key or not data[ts_key]:
                logger.error("No time series data in AlphaVantage response")
                return None

            # Parse the time series data
            ts_data = data[ts_key]
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for timestamp_str, candle in list(ts_data.items())[:limit]:
                try:
                    timestamp = datetime.strptime(
                        timestamp_str,
                        "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=self.tz)
                    
                    timestamps.append(timestamp)
                    opens.append(float(candle.get("1. open", 0)))
                    highs.append(float(candle.get("2. high", 0)))
                    lows.append(float(candle.get("3. low", 0)))
                    closes.append(float(candle.get("4. close", 0)))
                    volumes.append(float(candle.get("5. volume", 0)))
                except (ValueError, KeyError):
                    continue

            if not timestamps:
                logger.error("No valid candles parsed from AlphaVantage")
                return None

            df = pd.DataFrame({
                "timestamp": timestamps,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            })

            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            logger.info(
                "Received %d candles from AlphaVantage. "
                "Range: %s to %s",
                len(df),
                df.index[0] if len(df) > 0 else "N/A",
                df.index[-1] if len(df) > 0 else "N/A"
            )

            return df

        except requests.exceptions.Timeout:
            logger.error("AlphaVantage API request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection error: %s", e)
            return None
        except Exception as e:
            logger.exception("Error fetching AlphaVantage data: %s", e)
            return None


# ============================================================
# ANALYZER
# ============================================================

class TrinitySignalAnalyzer:
    """
    Implements the TRINITY three-stage setup:

        SWEEP
            ↓
        CONFIRMATION
            ↓
        ENTRY

    The analyzer intentionally favors fewer signals over forcing
    a setup when the required conditions are not present.
    """

    def __init__(self, data_fetcher: AlphaVantageDataFetcher):
        self.tz = TIMEZONE
        self.fetcher = data_fetcher

    def fetch_market_data(
        self,
        symbol: str = SYMBOL,
    ) -> Optional[pd.DataFrame]:

        df = self.fetcher.fetch_intraday(
            symbol,
            interval="1min",
            limit=100
        )

        if df is None or df.empty:
            logger.error("No market data available.")
            return None

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            logger.error("Missing columns: %s", missing)
            return None

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        ).copy()

        if len(df) < ATR_PERIOD + SWING_LOOKBACK + 10:
            logger.error(
                "Not enough candles. Received %s.",
                len(df)
            )
            return None

        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        logger.info(
            "Received %s valid candles. Latest: %s",
            len(df),
            df.index[-1]
        )

        return df

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = ATR_PERIOD
    ) -> pd.Series:

        previous_close = df["Close"].shift(1)

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - previous_close).abs()
        low_close = (df["Low"] - previous_close).abs()

        true_range = pd.concat(
            [high_low, high_close, low_close],
            axis=1
        ).max(axis=1)

        return true_range.rolling(
            window=period,
            min_periods=period
        ).mean()

    # --------------------------------------------------------
    # FVG / iFVG DETECTION (Visual only)
    # --------------------------------------------------------

    @staticmethod
    def detect_fvgs(
        df: pd.DataFrame,
        lookback: int = 50
    ) -> Dict[str, Any]:
        """
        Detect Fair Value Gaps and Internal FVGs.
        Returns nearby gaps for display only (doesn't affect trading).
        """

        if len(df) < 5:
            return {
                "fvgs": [],
                "ifvgs": [],
                "nearby": None
            }

        current_price = float(df["Close"].iloc[-1])
        recent = df.iloc[-lookback:]

        fvgs = []
        ifvgs = []

        for i in range(1, len(recent)):
            prev_candle = recent.iloc[i - 1]
            curr_candle = recent.iloc[i]

            prev_high = float(prev_candle["High"])
            prev_low = float(prev_candle["Low"])
            curr_open = float(curr_candle["Open"])
            curr_high = float(curr_candle["High"])
            curr_low = float(curr_candle["Low"])

            if curr_open > prev_high:
                gap_low = prev_high
                gap_high = curr_open
                gap_size = gap_high - gap_low
                fvgs.append({
                    "type": "BULLISH",
                    "low": gap_low,
                    "high": gap_high,
                    "size": gap_size,
                    "age": len(recent) - i
                })

            elif curr_high < prev_low:
                gap_high = prev_low
                gap_low = curr_high
                gap_size = gap_high - gap_low
                fvgs.append({
                    "type": "BEARISH",
                    "low": gap_low,
                    "high": gap_high,
                    "size": gap_size,
                    "age": len(recent) - i
                })

        nearby_fvg = None
        for fvg in fvgs:
            if fvg["type"] == "BULLISH":
                if current_price > fvg["high"]:
                    continue
            else:
                if current_price < fvg["low"]:
                    continue

            if nearby_fvg is None:
                nearby_fvg = fvg
                break

        return {
            "fvgs": fvgs[:5],
            "ifvgs": ifvgs,
            "nearby": nearby_fvg
        }

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    def detect_sweep(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        candle_index: int
    ) -> Optional[Sweep]:

        if candle_index < SWING_LOOKBACK + 1:
            return None

        current = df.iloc[candle_index]
        current_atr = atr.iloc[candle_index]

        if pd.isna(current_atr) or current_atr <= 0:
            return None

        structure_start = max(0, candle_index - SWING_LOOKBACK)
        structure_end = candle_index

        prior = df.iloc[structure_start:structure_end]

        if len(prior) < 5:
            return None

        prior_high = float(prior["High"].max())
        prior_low = float(prior["Low"].min())

        threshold = current_atr * SWEEP_ATR_MULTIPLIER

        bullish_sweep = (
            current["Low"] <= prior_low - threshold
            and current["Close"] > prior_low
        )

        bearish_sweep = (
            current["High"] >= prior_high + threshold
            and current["Close"] < prior_high
        )

        if bullish_sweep and not bearish_sweep:
            return Sweep(
                direction="LONG",
                sweep_level=prior_low,
                sweep_extreme=float(current["Low"]),
                candle_index=candle_index,
                atr=float(current_atr)
            )

        if bearish_sweep and not bullish_sweep:
            return Sweep(
                direction="SHORT",
                sweep_level=prior_high,
                sweep_extreme=float(current["High"]),
                candle_index=candle_index,
                atr=float(current_atr)
            )

        return None

    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------

    def check_confirmation(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        sweep: Sweep
    ) -> Optional[Confirmation]:

        start = sweep.candle_index + 1
        end = min(len(df), start + MAX_SETUP_AGE_BARS)

        if start >= end:
            return None

        for i in range(start, end):

            candle = df.iloc[i]
            candle_atr = atr.iloc[i]

            if pd.isna(candle_atr) or candle_atr <= 0:
                continue

            candle_range = float(
                candle["High"] - candle["Low"]
            )

            if candle_range <= 0:
                continue

            atr_multiple = candle_range / float(candle_atr)

            body = abs(
                float(candle["Close"])
                - float(candle["Open"])
            )

            close_location = (
                float(candle["Close"])
                - float(candle["Low"])
            ) / candle_range

            structure_start = max(0, i - STRUCTURE_LOOKBACK)
            prior = df.iloc[structure_start:i]

            if len(prior) < 5:
                continue

            prior_high = float(prior["High"].max())
            prior_low = float(prior["Low"].min())

            bullish_bos = (
                sweep.direction == "LONG"
                and candle["Close"] > prior_high
            )

            bearish_bos = (
                sweep.direction == "SHORT"
                and candle["Close"] < prior_low
            )

            bullish_displacement = (
                sweep.direction == "LONG"
                and candle_range >= (
                    DISPLACEMENT_ATR_MULTIPLIER * candle_atr
                )
                and candle["Close"] > candle["Open"]
                and close_location >= 0.70
                and body / candle_range >= 0.50
            )

            bearish_displacement = (
                sweep.direction == "SHORT"
                and candle_range >= (
                    DISPLACEMENT_ATR_MULTIPLIER * candle_atr
                )
                and candle["Close"] < candle["Open"]
                and close_location <= 0.30
                and body / candle_range >= 0.50
            )

            if bullish_bos or bullish_displacement:

                confirmation_type = (
                    "BOS + DISPLACEMENT"
                    if bullish_bos and bullish_displacement
                    else "BOS" if bullish_bos else "DISPLACEMENT"
                )

                return Confirmation(
                    direction="LONG",
                    confirmation_type=confirmation_type,
                    candle_index=i,
                    candle_high=float(candle["High"]),
                    candle_low=float(candle["Low"]),
                    candle_open=float(candle["Open"]),
                    candle_close=float(candle["Close"]),
                    candle_range=candle_range,
                    atr_multiple=atr_multiple
                )

            if bearish_bos or bearish_displacement:

                confirmation_type = (
                    "BOS + DISPLACEMENT"
                    if bearish_bos and bearish_displacement
                    else "BOS" if bearish_bos else "DISPLACEMENT"
                )

                return Confirmation(
                    direction="SHORT",
                    confirmation_type=confirmation_type,
                    candle_index=i,
                    candle_high=float(candle["High"]),
                    candle_low=float(candle["Low"]),
                    candle_open=float(candle["Open"]),
                    candle_close=float(candle["Close"]),
                    candle_range=candle_range,
                    atr_multiple=atr_multiple
                )

        return None

    # --------------------------------------------------------
    # IMPULSE / OTE
    # --------------------------------------------------------

    def calculate_ote_zone(
        self,
        df: pd.DataFrame,
        sweep: Sweep,
        confirmation: Confirmation
    ) -> Dict[str, float]:

        start = sweep.candle_index
        end = confirmation.candle_index + 1

        leg = df.iloc[start:end]

        impulse_high = float(leg["High"].max())
        impulse_low = float(leg["Low"].min())

        range_value = impulse_high - impulse_low

        if range_value <= 0:
            raise ValueError("Invalid impulse range.")

        if sweep.direction == "LONG":

            fib_50 = impulse_high - range_value * 0.50
            fib_705 = impulse_high - range_value * 0.705

            entry_low = min(fib_50, fib_705)
            entry_high = max(fib_50, fib_705)

        else:

            fib_50 = impulse_low + range_value * 0.50
            fib_705 = impulse_low + range_value * 0.705

            entry_low = min(fib_50, fib_705)
            entry_high = max(fib_50, fib_705)

        return {
            "impulse_high": impulse_high,
            "impulse_low": impulse_low,
            "entry_low": entry_low,
            "entry_high": entry_high
        }

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    def calculate_confidence(
        self,
        sweep: Sweep,
        confirmation: Confirmation,
        current_price: float
    ) -> int:

        score = 0

        score += 2

        if confirmation.confirmation_type == "BOS + DISPLACEMENT":
            score += 3
        elif confirmation.confirmation_type in {"BOS", "DISPLACEMENT"}:
            score += 2

        if confirmation.atr_multiple >= 2.0:
            score += 2
        elif confirmation.atr_multiple >= 1.5:
            score += 1

        if sweep.direction == "LONG":
            if current_price > sweep.sweep_level:
                score += 1
            if current_price > sweep.sweep_extreme:
                score += 1
        else:
            if current_price < sweep.sweep_level:
                score += 1
            if current_price < sweep.sweep_extreme:
                score += 1

        return int(min(10, max(1, score)))

    # --------------------------------------------------------
    # SIGNAL GENERATION
    # --------------------------------------------------------

    def generate_signal(
        self,
        df: pd.DataFrame
    ) -> Signal:

        generated_at = datetime.now(self.tz)

        current_price = float(df["Close"].iloc[-1])

        atr = self.calculate_atr(df)

        if pd.isna(atr.iloc[-1]):
            fvg_info = self.detect_fvgs(df)
            return Signal(
                valid=False,
                direction=None,
                confidence=0,
                current_price=current_price,
                sweep=None,
                confirmation=None,
                impulse_high=None,
                impulse_low=None,
                entry_low=None,
                entry_high=None,
                stop_loss=None,
                tp1=None,
                tp2=None,
                tp3=None,
                risk_points=None,
                risk_dollars_per_contract=None,
                generated_at=generated_at,
                fvg_info=fvg_info,
                reason="ATR unavailable."
            )

        latest_index = len(df) - 1
        earliest_index = max(
            SWING_LOOKBACK + 1,
            latest_index - SETUP_SCAN_BARS
        )

        candidates: List[Signal] = []

        for sweep_index in range(
            latest_index - MAX_SETUP_AGE_BARS,
            earliest_index - 1,
            -1
        ):

            if sweep_index <= 0:
                continue

            sweep = self.detect_sweep(df, atr, sweep_index)

            if sweep is None:
                continue

            confirmation = self.check_confirmation(df, atr, sweep)

            if confirmation is None:
                continue

            if confirmation.candle_index >= len(df):
                continue

            try:
                ote = self.calculate_ote_zone(df, sweep, confirmation)
            except Exception:
                continue

            confidence = self.calculate_confidence(
                sweep,
                confirmation,
                current_price
            )

            current_atr = float(atr.iloc[-1])

            if sweep.direction == "LONG":

                stop_loss = (
                    sweep.sweep_extreme
                    - current_atr * STOP_ATR_MULTIPLIER
                )

                reference_entry = (
                    ote["entry_low"]
                    + ote["entry_high"]
                ) / 2

                risk_points = (
                    reference_entry - stop_loss
                )

            else:

                stop_loss = (
                    sweep.sweep_extreme
                    + current_atr * STOP_ATR_MULTIPLIER
                )

                reference_entry = (
                    ote["entry_low"]
                    + ote["entry_high"]
                ) / 2

                risk_points = (
                    stop_loss - reference_entry
                )

            if risk_points <= 0:
                continue

            if risk_points > MAX_STOP_LOSS_POINTS:
                logger.debug(
                    "Setup rejected: stop loss too wide "
                    "(%s pts > %s pt limit)",
                    round(risk_points, 2),
                    MAX_STOP_LOSS_POINTS
                )
                continue

            risk_dollars = (
                risk_points
                * NQ_DOLLARS_PER_POINT
            )

            if sweep.direction == "LONG":

                tp1 = reference_entry + (
                    risk_points * TP1_R
                )
                tp2 = reference_entry + (
                    risk_points * TP2_R
                )
                tp3 = reference_entry + (
                    risk_points * TP3_R
                )

            else:

                tp1 = reference_entry - (
                    risk_points * TP1_R
                )
                tp2 = reference_entry - (
                    risk_points * TP2_R
                )
                tp3 = reference_entry - (
                    risk_points * TP3_R
                )

            valid = confidence >= MIN_CONFIDENCE

            fvg_info = self.detect_fvgs(df)

            candidate = Signal(
                valid=valid,
                direction=sweep.direction,
                confidence=confidence,
                current_price=current_price,
                sweep=sweep,
                confirmation=confirmation,
                impulse_high=ote["impulse_high"],
                impulse_low=ote["impulse_low"],
                entry_low=ote["entry_low"],
                entry_high=ote["entry_high"],
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                risk_points=risk_points,
                risk_dollars_per_contract=risk_dollars,
                generated_at=generated_at,
                fvg_info=fvg_info,
                reason=(
                    None
                    if valid
                    else (
                        f"Confidence {confidence}/10 "
                        f"is below minimum {MIN_CONFIDENCE}/10."
                    )
                )
            )

            candidates.append(candidate)

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.valid
        ]

        if valid_candidates:
            valid_candidates.sort(
                key=lambda x: (
                    x.confirmation.candle_index
                    if x.confirmation
                    else -1
                ),
                reverse=True
            )

            return valid_candidates[0]

        fvg_info = self.detect_fvgs(df)

        return Signal(
            valid=False,
            direction=None,
            confidence=0,
            current_price=current_price,
            sweep=None,
            confirmation=None,
            impulse_high=None,
            impulse_low=None,
            entry_low=None,
            entry_high=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            tp3=None,
            risk_points=None,
            risk_dollars_per_contract=None,
            generated_at=generated_at,
            fvg_info=fvg_info,
            reason=(
                "No completed TRINITY setup met "
                f"the {MIN_CONFIDENCE}/10 confidence threshold."
            )
        )

    # --------------------------------------------------------
    # MESSAGE FORMAT
    # --------------------------------------------------------

    def format_signal_message(
        self,
        signal: Signal
    ) -> str:

        timestamp = signal.generated_at.strftime("%I:%M %p ET")
        price = signal.current_price

        if not signal.valid:

            return (
                "📊 **TRINITY NQ DAILY ANALYSIS**\n\n"
                f"**Time:** {timestamp}\n"
                f"**Current Price:** `{price:,.2f}`\n\n"
                "🟡 **STATUS: NO VALID SETUP**\n\n"
                f"{signal.reason}\n\n"
                "TRINITY requires:\n"
                "🔵 Sweep\n"
                "🟢 Confirmation\n"
                "🟡 OTE Entry\n\n"
                "No trade is being called."
            )

        sweep = signal.sweep
        confirmation = signal.confirmation

        direction_emoji = (
            "🟢" if signal.direction == "LONG" else "🔴"
        )

        return (
            f"{direction_emoji} **TRINITY NQ "
            f"{signal.direction} SETUP**\n\n"

            f"**Time:** {timestamp}\n"
            f"**Current Price:** `{price:,.2f}`\n\n"

            "🔵 **STAGE 1 — SWEEP**\n"
            f"Type: `{signal.direction}`\n"
            f"Swept Level: `{sweep.sweep_level:,.2f}`\n"
            f"Sweep Extreme: `{sweep.sweep_extreme:,.2f}`\n\n"

            "🟢 **STAGE 2 — CONFIRMATION**\n"
            f"Type: `{confirmation.confirmation_type}`\n"
            f"Displacement: `{confirmation.atr_multiple:.2f}x ATR`\n\n"

            "🟡 **STAGE 3 — ENTRY**\n"
            f"OTE Zone: `{signal.entry_low:,.2f}"
            f" – {signal.entry_high:,.2f}`\n\n"

            f"**STOP:** `{signal.stop_loss:,.2f}`\n"
            f"Risk: `{signal.risk_points:.2f}` pts "
            f"(${signal.risk_dollars_per_contract:,.2f}"
            f"/contract)\n\n"

            f"**TP1:** `{signal.tp1:,.2f}` — 1R\n"
            f"**TP2:** `{signal.tp2:,.2f}` — 1.5R\n"
            f"**TP3:** `{signal.tp3:,.2f}` — 2R\n\n"

            f"**CONFIDENCE:** `{signal.confidence}/10`\n\n"

            "⚠️ **EXECUTION:** Wait for price to enter "
            "the OTE zone and confirm rejection before entering."
        ) + self._format_fvg_info(signal)

    def _format_fvg_info(self, signal: Signal) -> str:
        """Format FVG info for display (visual only)"""

        if not signal.fvg_info or not signal.fvg_info.get("nearby"):
            return ""

        nearby = signal.fvg_info["nearby"]

        return (
            "\n\n📊 **ADDITIONAL CONTEXT (Visual Reference)**\n"
            f"Nearby FVG: {nearby['type']}\n"
            f"Zone: `{nearby['low']:,.2f} – {nearby['high']:,.2f}`\n"
            f"(FVG info for reference only — doesn't affect setup logic)"
        )


# ============================================================
# DISCORD BOT
# ============================================================

class TrinityBot(commands.Bot):

    def __init__(self, data_fetcher: AlphaVantageDataFetcher):

        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix="!",
            intents=intents
        )

        self.analyzer = TrinitySignalAnalyzer(data_fetcher)
        self.last_post_date: Optional[date] = None

    async def setup_hook(self):

        self.market_scanner.start()

    async def on_ready(self):

        logger.info(
            "Logged in as %s (%s)",
            self.user,
            self.user.id
        )

        logger.info(
            "TRINITY configured for %s (24/7 scanning, unlimited signals)",
            SYMBOL
        )

        logger.info(
            "Max stop loss: %.1f points",
            MAX_STOP_LOSS_POINTS
        )

    # --------------------------------------------------------
    # 24/7 Market Scanner
    # --------------------------------------------------------

    @tasks.loop(minutes=1)
    async def market_scanner(self):
        """Continuously scan for TRINITY setups 24/7 - unlimited signals"""

        await self.check_for_signal()

    @market_scanner.before_loop
    async def before_market_scanner(self):

        await self.wait_until_ready()

    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    async def get_signal_channel(
        self
    ) -> Optional[discord.abc.Messageable]:

        if not CHANNEL_ID:
            logger.error(
                "CHANNEL_ID is not configured."
            )
            return None

        try:
            channel_id = int(CHANNEL_ID)
        except ValueError:
            logger.error(
                "CHANNEL_ID must be numeric."
            )
            return None

        channel = self.get_channel(
            channel_id
        )

        if channel is not None:
            return channel

        try:
            channel = await self.fetch_channel(
                channel_id
            )
            return channel
        except Exception as exc:
            logger.error(
                "Could not fetch channel %s: %s",
                channel_id,
                exc
            )
            return None

    # --------------------------------------------------------
    # Check For & Post Signal
    # --------------------------------------------------------

    async def check_for_signal(self):
        """Check market and post signal if valid setup found"""

        logger.debug("Scanning market for TRINITY setups...")

        try:

            df = self.analyzer.fetch_market_data(
                SYMBOL
            )

            if df is None:
                logger.error(
                    "Signal aborted: market data unavailable."
                )
                return

            signal = self.analyzer.generate_signal(
                df
            )

            if not signal.valid:
                logger.debug(
                    "No valid setup: %s",
                    signal.reason
                )
                return

            message = (
                self.analyzer.format_signal_message(
                    signal
                )
            )

            channel = await self.get_signal_channel()

            if channel is None:
                logger.error(
                    "Signal aborted: Discord channel unavailable."
                )
                return

            await channel.send(
                message
            )

            logger.info(
                "TRINITY signal posted to Discord."
            )

        except Exception as exc:

            logger.exception(
                "Error generating/posting signal: %s",
                exc
            )

    # --------------------------------------------------------
    # Manual Signal
    # --------------------------------------------------------

    @commands.command(
        name="signal"
    )
    async def manual_signal(
        self,
        ctx: commands.Context
    ):

        logger.info(
            "Manual signal requested by %s",
            ctx.author
        )

        await ctx.send(
            "⏳ **TRINITY is analyzing NQ...**"
        )

        try:

            df = self.analyzer.fetch_market_data(
                SYMBOL
            )

            if df is None:
                await ctx.send(
                    "❌ Unable to retrieve NQ market data."
                )
                return

            signal = self.analyzer.generate_signal(
                df
            )

            message = (
                self.analyzer.format_signal_message(
                    signal
                )
            )

            await ctx.send(
                message
            )

        except Exception as exc:

            logger.exception(
                "Manual signal error: %s",
                exc
            )

            await ctx.send(
                "❌ An internal error occurred while "
                "generating the signal. Check bot logs."
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    errors = []

    if not DISCORD_TOKEN:
        errors.append(
            "DISCORD_TOKEN is missing."
        )

    if not CHANNEL_ID:
        errors.append(
            "CHANNEL_ID is missing."
        )

    if not ALPHAVANTAGE_API_KEY:
        errors.append(
            "ALPHAVANTAGE_API_KEY is missing."
        )

    if not 0 <= POST_HOUR <= 23:
        errors.append(
            "POST_HOUR must be between 0 and 23."
        )

    if not 0 <= POST_MINUTE <= 59:
        errors.append(
            "POST_MINUTE must be between 0 and 59."
        )

    if not 1 <= MIN_CONFIDENCE <= 10:
        errors.append(
            "MIN_CONFIDENCE must be between 1 and 10."
        )

    if MAX_TRADES_PER_DAY < 1:
        errors.append(
            "MAX_TRADES_PER_DAY must be at least 1."
        )

    if errors:

        for error in errors:
            logger.error(error)

        raise ValueError(
            "Invalid configuration. "
            "Fix the .env file before starting."
        )


# ============================================================
# ENTRY POINT
# ============================================================

async def main():

    validate_config()

    av_fetcher = AlphaVantageDataFetcher(
        ALPHAVANTAGE_API_KEY
    )

    bot = TrinityBot(av_fetcher)

    try:

        logger.info(
            "🚀 TRINITY Signal Bot (AlphaVantage) starting..."
        )

        logger.info(
            "Symbol: %s",
            SYMBOL
        )

        logger.info(
            "Daily post: %02d:%02d ET",
            POST_HOUR,
            POST_MINUTE
        )

        logger.info(
            "Minimum confidence: %d/10",
            MIN_CONFIDENCE
        )

        await bot.start(
            DISCORD_TOKEN
        )

    finally:

        if not bot.is_closed():
            await bot.close()


if __name__ == "__main__":

    try:
        import asyncio

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "TRINITY bot stopped."
        )

    except Exception as exc:

        logger.exception(
            "Fatal error: %s",
            exc
        )
