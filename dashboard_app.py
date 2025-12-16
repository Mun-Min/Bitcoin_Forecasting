import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np

# Import loader
from BTC_Data_Pipline import BTCDataLoader

# -------------------------
# Load data using data pipeline class
# -------------------------
@st.cache_data(show_time=True, show_spinner=True)
def load_data():
    loader = BTCDataLoader(kaggle_dataset="mczielinski/bitcoin-historical-data")
    df_hourly, df_daily = loader.load_and_clean(save_dir=None)
    return df_hourly, df_daily

df_hourly, df_daily = load_data()

st.set_page_config(
    page_title="BTC Dashboard",
    page_icon="💰", 
    layout="wide" 
)

# -------------------------
# Sidebar navigation
# -------------------------
with st.sidebar:
    selected = option_menu(
        "Bitcoin Dashboard",
        ["Visualizations", "Forecasting"],
        icons=["bar-chart", "graph-up"],
        menu_icon="currency-bitcoin",
        default_index=0,
        orientation="vertical",
    )

# -------------------------
# PAGE 1 — VISUALIZATIONS
# 
# -------------------------
if selected == "Visualizations":
    st.title("📊 Bitcoin Market Visualizations")
    st.subheader("Daily Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_daily.index,
            open=df_daily["Open"],
            high=df_daily["High"],
            low=df_daily["Low"],
            close=df_daily["Close"],
            name="OHLC"
        )
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Daily Volume (BTC) + Estimated USD volume (millions)
    # -------------------------
    vol_candidates = [c for c in df_daily.columns if "vol" in c.lower()]
    if not vol_candidates:
        st.error("No volume column found in daily dataset.")
    else:
        vol_col = "Volume" if "Volume" in df_daily.columns else vol_candidates[0]
        btc_vol = df_daily[vol_col].astype(float).copy()
        usd_est = btc_vol * df_daily["Close"]
        btc_vol_m = btc_vol / 1e6
        usd_est_m = usd_est / 1e6

        fig_vol = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig_vol.add_trace(
            go.Bar(
                x=df_daily.index,
                y=btc_vol_m,
                name=f"BTC Volume ({vol_col}) [millions BTC]",
                marker_color=np.where(df_daily["Close"] >= df_daily["Open"], "green", "red"),
                hovertemplate="%{x}<br>BTC Volume: %{customdata[0]:,.0f} BTC<br>%{y:.3f}M BTC<extra></extra>",
                customdata=np.stack([btc_vol.values], axis=-1),
            ),
            row=1, col=1, secondary_y=False
        )
        fig_vol.add_trace(
            go.Scatter(
                x=df_daily.index,
                y=usd_est_m,
                mode="lines",
                name="Estimated USD Volume [millions USD]",
                line=dict(width=2, color="royalblue"),
                hovertemplate="%{x}<br>Estimated USD Volume: %{customdata[0]:,.0f} USD<br>%{y:.2f}M USD<extra></extra>",
                customdata=np.stack([usd_est.values], axis=-1),
            ),
            row=1, col=1, secondary_y=True
        )
        fig_vol.update_layout(
            height=350,
            title_text="BTC Volume (bars, millions BTC) and Estimated USD Volume (line, millions USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig_vol.update_yaxes(title_text=f"BTC Volume (millions {vol_col})", secondary_y=False)
        fig_vol.update_yaxes(title_text="Estimated USD Volume (millions USD)", secondary_y=True)
        st.plotly_chart(fig_vol, use_container_width=True)

    st.info("Data pulled automatically through BTCDataLoader → cleaned → resampled → visualized.")
    st.info("Data Source: [Kaggle - Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)")

# -------------------------
# PAGE 2 — FORECASTING 
# -------------------------
if selected == "Forecasting":
    st.title("📈 Bitcoin Forecasting — Naive & SMA")
    st.write("Forecasting uses **daily closing prices**. Simple baseline models: Naive (last value or with drift) and SMA-extension (with optional momentum). ")

    df = df_daily.copy()
    # Ensure datetime index, daily frequency, forward-fill missing days
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if "Close" not in df.columns:
        st.error("Daily dataset has no 'Close' column — cannot forecast.")
    else:
        st.subheader("Historical Close Price")
        st.line_chart(df["Close"], height=300)

        # Controls
        st.subheader("Forecast settings")
        days_ahead = int(st.number_input(
            "Forecast days ahead:",
            1, 365, 14,
            help="Number of calendar days to forecast into the future. Short horizons tend to be more accurate."
        ))

        model_choice = st.selectbox(
            "Model:",
            ["Naive (last value / last value + drift)", "SMA-extension (optionally + momentum)"],
            help=("Choose a baseline forecasting method:\n\n"
                  "- Naive: repeats the last observed value (optionally with a small drift computed from recent daily changes). "
                  "Good as a simple benchmark.\n\n"
                  "- SMA-extension: uses the last Simple Moving Average (SMA) value as the baseline and optionally extends it "
                  "by adding a small momentum term (slope) computed from recent SMA behaviour.")
        )

        # short caption -> visible explanation under selector
        st.caption("Tip: Start with a short holdout (e.g., 14–30 days) to see quick MAE feedback. Use SMA window to control smoothing; momentum amplifies recent trends.")

        test_size_days = int(st.slider(
            "Holdout size (days) for quick validation",
            7, 90, 30,
            help="Number of most recent days held out for quick out-of-sample validation. Used only to compute MAE."
        ))

        # Series prepared at daily frequency
        series = df["Close"].asfreq('D').fillna(method="ffill")

        # train/test
        train = series.iloc[:-test_size_days]
        test = series.iloc[-test_size_days:]
        st.markdown(f"Training on **{train.index[0].date()} → {train.index[-1].date()}** ({len(train)} days). "
                    f"Testing on **{test.index[0].date()} → {test.index[-1].date()}** ({len(test)} days).")

        # plotting helper
        def plot_forecast(train_series, test_series, forecast_index, forecast_mean, conf_lower=None, conf_upper=None, title="Forecast"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_series.index, y=train_series.values, mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=test_series.index, y=test_series.values, mode='lines', name='Test', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast', line=dict(width=2)))
            if conf_lower is not None and conf_upper is not None:
                fig.add_trace(go.Scatter(
                    x=list(forecast_index) + list(forecast_index[::-1]),
                    y=list(conf_upper) + list(conf_lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='CI'
                ))
            fig.update_layout(height=480, title=title, xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig, use_container_width=True)

        # utility: compute MAE
        from sklearn.metrics import mean_absolute_error

        # utility: compute simple volatility-based CI for multi-day horizon
        def horizon_ci_from_returns(series_train, forecast_mean, alpha=1.96):
            """
            Use historical daily log-return std to estimate CI for multi-day forecast:
            sigma_h = sigma_daily * sqrt(h)
            then +/- alpha * sigma_h applied in price space via multiplicative log-normal approx.
            Returns lower, upper arrays aligned to forecast_mean index.
            """
            # use log returns to model multiplicative uncertainty
            logr = np.log(series_train / series_train.shift(1)).dropna()
            sigma_daily = logr.std()
            if pd.isna(sigma_daily) or sigma_daily == 0:
                # fallback to small constant
                sigma_daily = 1e-4
            horizons = np.arange(1, len(forecast_mean) + 1)
            sigma_h = sigma_daily * np.sqrt(horizons)
            # forecast_mean may be array; apply log-normal CI: lower = mean * exp(-alpha*sigma_h), upper = mean * exp(alpha*sigma_h)
            lower = forecast_mean * np.exp(-alpha * sigma_h)
            upper = forecast_mean * np.exp(alpha * sigma_h)
            return lower, upper

        # run chosen model
        if model_choice.startswith("Naive"):
            st.markdown("**Naive model options** — constant last value or include drift (average daily change).")
            use_drift = st.checkbox(
                "Include drift (average daily change over last N days)",
                value=True,
                help="If checked, the forecast will add the average daily change observed over the selected drift window to the last value (linear drift)."
            )
            drift_window = int(st.number_input(
                "Drift window (days)",
                1, 90, 14,
                help="Number of recent days used to compute average daily change (drift). A small window responds faster to recent movement."
            )) if use_drift else None

            last_val = train.iloc[-1]
            future_index = pd.date_range(series.index[-1], periods=days_ahead + 1, freq='D')[1:]

            if not use_drift:
                forecast_vals = np.repeat(last_val, len(future_index))
                pred_test = np.repeat(last_val, len(test))
            else:
                # compute average daily change (not percent) over drift_window on train tail
                drift_window = max(1, drift_window)
                recent = train.iloc[-drift_window:]
                avg_daily_change = (recent.diff().dropna().mean())  # this is scalar (pandas Series -> value)
                # If recent.diff().mean() is a Series, extract value
                if hasattr(avg_daily_change, 'item'):
                    avg_daily_change = avg_daily_change.item()
                # build forecast by adding avg_daily_change cumulatively
                forecast_vals = np.array([last_val + avg_daily_change * (i + 1) for i in range(len(future_index))])
                pred_test = np.array([train.iloc[-1] + avg_daily_change * (i + 1) for i in range(len(test))])

            mae = mean_absolute_error(test.values, pred_test)
            st.write(f"MAE on holdout: {mae:,.4f}")

            # confidence intervals via historical returns
            conf_lower, conf_upper = horizon_ci_from_returns(train, forecast_vals)

            plot_forecast(train, test, future_index, forecast_vals, conf_lower=conf_lower, conf_upper=conf_upper,
                          title=f"Naive Forecast {'with drift' if use_drift else '(last value)'}")

            # prepare download
            out_df = pd.DataFrame({
                "forecast_date": future_index,
                "forecast": forecast_vals,
                "ci_lower": conf_lower,
                "ci_upper": conf_upper
            }).set_index("forecast_date")
            csv = out_df.to_csv().encode('utf-8')
            st.download_button(
                "Download forecast CSV",
                csv,
                file_name="btc_naive_forecast.csv",
                mime="text/csv",
                help="Download the forecast as a CSV. Contains forecast, lower CI, upper CI."
            )

        elif model_choice.startswith("SMA"):
            st.markdown("**SMA-extension** — forecast = last SMA, optionally add a small momentum/trend component.")
            sma_window = int(st.slider(
                "SMA window (days)",
                3, 300, 30,
                help="Number of days used to compute the Simple Moving Average (SMA). Larger windows produce smoother SMA."
            ))
            sma_series = series.rolling(window=sma_window, min_periods=1).mean()
            last_sma = sma_series.iloc[-1]

            add_momentum = st.checkbox(
                "Add momentum (slope) to extend SMA forecast",
                value=True,
                help="If checked, compute a recent slope from SMA values and apply it to extend the SMA forward."
            )
            momentum_window = int(st.number_input(
                "Momentum window (days, used to compute slope)",
                3, 180, 14,
                help="Number of SMA days used to estimate the slope. Short windows react faster but are noisier."
            )) if add_momentum else None
            momentum_strength = float(st.slider(
                "Momentum multiplier (1.0 = full slope)",
                0.0, 3.0, 1.0,
                help="Scale the estimated slope. Use <1 to dampen, >1 to amplify recent trend."
            ))

            future_index = pd.date_range(series.index[-1], periods=days_ahead + 1, freq='D')[1:]

            if not add_momentum:
                forecast_vals = np.repeat(last_sma, len(future_index))
                # test prediction (naive: repeat last_sma)
                pred_test = np.repeat(last_sma, len(test))
            else:
                # compute linear slope (price change per day) over momentum_window from train tail (use SMA values)
                momentum_window = max(3, momentum_window)
                slope_series = sma_series.dropna()
                if len(slope_series) < momentum_window:
                    slope_series_used = slope_series
                else:
                    slope_series_used = slope_series.iloc[-momentum_window:]
                # fit simple linear regression (slope)
                x = np.arange(len(slope_series_used))
                y = slope_series_used.values
                # handle degenerate cases
                if len(x) < 2 or np.allclose(y, y[0]):
                    slope_per_day = 0.0
                else:
                    A = np.vstack([x, np.ones_like(x)]).T
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    slope_per_day = m  # price change per day
                slope_per_day *= momentum_strength
                # forecast: last_sma plus cumulative slope
                forecast_vals = np.array([last_sma + slope_per_day * (i + 1) for i in range(len(future_index))])
                # create test preds by applying slope from end of train
                pred_test = np.array([last_sma + slope_per_day * (i + 1) for i in range(len(test))])

            mae = mean_absolute_error(test.values, pred_test)
            st.write(f"SMA window: {sma_window} days — MAE on holdout: {mae:,.4f}")
            if add_momentum:
                st.write(f"Momentum window: {momentum_window} days — slope/day (applied): {slope_per_day:.6f}")

            # confidence intervals using historical returns
            conf_lower, conf_upper = horizon_ci_from_returns(train, forecast_vals)

            plot_forecast(train, test, future_index, forecast_vals, conf_lower=conf_lower, conf_upper=conf_upper,
                          title=f"SMA-extension Forecast (window={sma_window}{', + momentum' if add_momentum else ''})")

            # download
            out_df = pd.DataFrame({
                "forecast_date": future_index,
                "forecast": forecast_vals,
                "ci_lower": conf_lower,
                "ci_upper": conf_upper
            }).set_index("forecast_date")
            csv = out_df.to_csv().encode('utf-8')
            st.download_button(
                "Download forecast CSV",
                csv,
                file_name="btc_sma_forecast.csv",
                mime="text/csv",
                help="Download the SMA forecast as CSV (forecast, lower CI, upper CI)."
            )
