"""
Multi-Horizon Demand Forecasting & Conformal Uncertainty Quantification Engine.

Provides:
1. Daily transaction time-series reconstruction and rolling velocity indicators.
2. Classical/Robust time-series decomposition: Baseline Level, Trend, Day-of-Week Seasonality.
3. Multi-horizon forward demand forecasting (7-day, 14-day, 30-day, 60-day) with category breakouts.
4. Conformal prediction intervals: empirical distribution-free 80% & 95% uncertainty bounds.
5. Walk-forward backtesting evaluation: MAE, RMSE, MAPE, WAPE, and Conformal Coverage Ratio.
6. Rolling Z-score demand spike and operational anomaly detection.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics import RetailAnalyticsEngine


class MultiHorizonDemandForecaster:
    """
    Enterprise multi-horizon retail demand forecasting engine with conformal uncertainty
    intervals and statistical surge anomaly detection.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.analytics = RetailAnalyticsEngine(data_path=data_path)
        self.df = self.analytics.df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.categories = sorted(self.df["Product Category"].unique().tolist())

    def get_daily_demand_series(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts and resamples a continuous daily time-series of demand units,
        revenue, and rolling velocity indicators.
        """
        data = self.df if category is None else self.df[self.df["Product Category"] == category]

        if len(data) == 0:
            raise ValueError(f"Category '{category}' has no transaction records.")

        # Aggregate daily
        daily = (
            data.groupby("Date")
            .agg(
                units=("Quantity", "sum"),
                revenue=("Total Amount", "sum"),
                orders=("Transaction ID", "count"),
                avg_price=("Price per Unit", "mean"),
            )
            .reset_index()
        )

        # Ensure continuous daily date index over the full catalog date range
        daily = daily.set_index("Date")
        min_date = self.df["Date"].min()
        max_date = self.df["Date"].max()
        idx = pd.date_range(min_date, max_date, freq="D")
        daily = daily.reindex(idx)
        daily["units"] = daily["units"].fillna(0.0)
        daily["revenue"] = daily["revenue"].fillna(0.0)
        daily["orders"] = daily["orders"].fillna(0)
        daily["avg_price"] = daily["avg_price"].ffill().bfill()
        daily = daily.reset_index().rename(columns={"index": "Date"})

        daily["day_of_week"] = daily["Date"].dt.day_name()
        daily["day_of_week_num"] = daily["Date"].dt.dayofweek
        daily["month"] = daily["Date"].dt.month

        # Rolling velocities
        daily["rolling_7d_mean"] = daily["units"].rolling(window=7, min_periods=1).mean()
        daily["rolling_14d_mean"] = daily["units"].rolling(window=14, min_periods=1).mean()
        daily["rolling_30d_mean"] = daily["units"].rolling(window=30, min_periods=1).mean()
        daily["rolling_14d_std"] = daily["units"].rolling(window=14, min_periods=1).std().fillna(0.0)

        return daily

    def decompose_demand(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Decomposes daily demand into Baseline Level, Linear Trend, Day-of-Week Seasonality,
        and Residual error variance.
        """
        daily = self.get_daily_demand_series(category=category)
        n = len(daily)
        t = np.arange(n)
        y = daily["units"].values

        # 1. Baseline Level
        level = float(np.mean(y))

        # 2. Linear Trend Slope
        if n > 1 and np.var(t) > 0:
            slope, intercept, r_value, p_value, _ = stats.linregress(t, y)
        else:
            slope, intercept, r_value, p_value = 0.0, level, 0.0, 1.0

        # 3. Day-of-week seasonality factors (multipliers relative to overall mean)
        dow_agg = daily.groupby("day_of_week_num")["units"].mean()
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_factors = {}
        for dow_num, name in enumerate(dow_names):
            val = float(dow_agg.get(dow_num, level))
            factor = (val / level) if level > 0 else 1.0
            dow_factors[name] = round(factor, 3)

        # 4. Residual variance
        fitted = np.array(
            [
                (intercept + slope * t_i) * dow_factors[daily.loc[t_i, "day_of_week"]]
                for t_i in range(n)
            ]
        )
        residuals = y - fitted
        res_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

        return {
            "category": category if category else "All Categories",
            "total_days_observed": n,
            "baseline_level_units": round(level, 3),
            "trend_daily_slope": round(float(slope), 5),
            "trend_annualized_slope": round(float(slope) * 365.0, 2),
            "trend_r_squared": round(float(r_value**2), 4),
            "trend_p_value": round(float(p_value), 4),
            "day_of_week_seasonality": dow_factors,
            "residual_std": round(res_std, 3),
        }

    def _fit_model(self, series: pd.DataFrame) -> Tuple[float, float, Dict[str, float], np.ndarray]:
        """
        Fits trend and seasonal parameters on a given historical slice.
        """
        n = len(series)
        t = np.arange(n)
        y = series["units"].values
        level = float(np.mean(y)) if n > 0 else 0.0

        if n > 1 and np.var(t) > 0:
            slope, intercept, _, _, _ = stats.linregress(t, y)
        else:
            slope, intercept = 0.0, level

        dow_agg = series.groupby("day_of_week_num")["units"].mean()
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_factors = {}
        for dow_num, name in enumerate(dow_names):
            val = float(dow_agg.get(dow_num, level))
            factor = (val / level) if level > 0 else 1.0
            dow_factors[name] = factor

        fitted = np.array(
            [
                max(0.0, (intercept + slope * t_i) * dow_factors[series.loc[t_i, "day_of_week"]])
                for t_i in range(n)
            ]
        )
        residuals = y - fitted
        return slope, intercept, dow_factors, residuals

    def forecast_demand(
        self,
        horizon_days: int = 30,
        category: Optional[str] = None,
        alpha_conformal: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Generates forward demand forecasts for `horizon_days` with distribution-free
        conformal uncertainty prediction intervals (80% and 95% coverage).
        """
        if horizon_days <= 0 or horizon_days > 180:
            raise ValueError(f"Horizon days must be in [1, 180], got {horizon_days}")

        daily = self.get_daily_demand_series(category=category)
        n = len(daily)
        slope, intercept, dow_factors, residuals = self._fit_model(daily)

        # Conformal residuals calibration
        abs_res = np.abs(residuals)
        q80 = float(np.percentile(abs_res, 80)) if len(abs_res) > 0 else 1.0
        q95 = float(np.percentile(abs_res, 95)) if len(abs_res) > 0 else 2.0

        last_date = daily["Date"].max()
        avg_price = float(daily["avg_price"].iloc[-1])

        projections = []
        cum_units = 0.0
        cum_units_lower95 = 0.0
        cum_units_upper95 = 0.0

        for h in range(1, horizon_days + 1):
            future_date = last_date + pd.Timedelta(days=h)
            dow_name = future_date.day_name()
            dow_factor = dow_factors.get(dow_name, 1.0)

            t_val = n + h - 1
            pred_units = max(0.0, (intercept + slope * t_val) * dow_factor)
            lower_80 = max(0.0, pred_units - q80)
            upper_80 = pred_units + q80
            lower_95 = max(0.0, pred_units - q95)
            upper_95 = pred_units + q95

            projected_rev = pred_units * avg_price

            cum_units += pred_units
            cum_units_lower95 += lower_95
            cum_units_upper95 += upper_95

            projections.append(
                {
                    "day_ahead": h,
                    "date": future_date.strftime("%Y-%m-%d"),
                    "day_of_week": dow_name,
                    "predicted_units": round(pred_units, 2),
                    "lower_bound_80": round(lower_80, 2),
                    "upper_bound_80": round(upper_80, 2),
                    "lower_bound_95": round(lower_95, 2),
                    "upper_bound_95": round(upper_95, 2),
                    "projected_revenue": round(projected_rev, 2),
                }
            )

        return {
            "category": category if category else "All Categories",
            "forecast_horizon_days": horizon_days,
            "historical_days_evaluated": n,
            "conformal_q80_margin": round(q80, 2),
            "conformal_q95_margin": round(q95, 2),
            "total_projected_units": round(cum_units, 1),
            "projected_units_95_lower": round(cum_units_lower95, 1),
            "projected_units_95_upper": round(cum_units_upper95, 1),
            "total_projected_revenue": round(cum_units * avg_price, 2),
            "avg_unit_price": round(avg_price, 2),
            "daily_projections": projections,
        }

    def evaluate_forecast_accuracy(
        self,
        test_window_days: int = 45,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes holdout walk-forward backtesting to quantify empirical forecast accuracy:
        MAE, RMSE, MAPE, WAPE, and Conformal 95% Coverage Rate.
        """
        daily = self.get_daily_demand_series(category=category)
        n = len(daily)

        if test_window_days >= n - 30:
            raise ValueError(
                f"Test window ({test_window_days}) too large for dataset of {n} days. Must leave at least 30 training days."
            )

        train = daily.iloc[:-test_window_days].copy().reset_index(drop=True)
        test = daily.iloc[-test_window_days:].copy().reset_index(drop=True)

        slope, intercept, dow_factors, train_residuals = self._fit_model(train)
        abs_train_res = np.abs(train_residuals)
        q95 = float(np.percentile(abs_train_res, 95)) if len(abs_train_res) > 0 else 2.0

        n_train = len(train)
        preds = []
        inside_95_count = 0
        actuals = test["units"].values

        for i, row in test.iterrows():
            t_val = n_train + i
            dow_name = row["day_of_week"]
            pred = max(0.0, (intercept + slope * t_val) * dow_factors.get(dow_name, 1.0))
            preds.append(pred)

            lower_95 = max(0.0, pred - q95)
            upper_95 = pred + q95
            if lower_95 <= row["units"] <= upper_95:
                inside_95_count += 1

        preds = np.array(preds)
        errors = actuals - preds
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        sum_actuals = float(np.sum(actuals))
        wape = (float(np.sum(abs_errors)) / sum_actuals * 100.0) if sum_actuals > 0 else 0.0

        # Avoid div by zero in MAPE
        non_zero_mask = actuals > 0
        if np.sum(non_zero_mask) > 0:
            mape = float(np.mean(abs_errors[non_zero_mask] / actuals[non_zero_mask])) * 100.0
        else:
            mape = 0.0

        coverage_rate = (inside_95_count / len(test)) * 100.0

        return {
            "category": category if category else "All Categories",
            "test_window_days": test_window_days,
            "train_days": n_train,
            "mae_units": round(mae, 3),
            "rmse_units": round(rmse, 3),
            "mape_percentage": round(mape, 2),
            "wape_percentage": round(wape, 2),
            "conformal_95_coverage_percentage": round(coverage_rate, 2),
            "conformal_calibration_margin": round(q95, 2),
        }

    def detect_demand_anomalies(
        self,
        category: Optional[str] = None,
        z_threshold: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """
        Identifies demand spikes and operational velocity outliers using a
        rolling 14-day Z-score indicator.
        """
        daily = self.get_daily_demand_series(category=category)
        anomalies = []

        for _, row in daily.iterrows():
            roll_mean = row["rolling_14d_mean"]
            roll_std = row["rolling_14d_std"]
            actual = row["units"]

            if roll_std > 1e-4:
                z = (actual - roll_mean) / roll_std
            else:
                z = 0.0

            if abs(z) >= z_threshold:
                if z >= 3.0:
                    severity = "CRITICAL_SPIKE"
                elif z >= z_threshold:
                    severity = "MODERATE_SPIKE"
                else:
                    severity = "DEMAND_TROUGH"

                anomalies.append(
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "category": category if category else "All Categories",
                        "actual_units": int(actual),
                        "expected_14d_mean_units": round(float(roll_mean), 2),
                        "rolling_std_units": round(float(roll_std), 2),
                        "z_score": round(float(z), 2),
                        "anomaly_severity": severity,
                        "day_of_week": row["day_of_week"],
                    }
                )

        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies
