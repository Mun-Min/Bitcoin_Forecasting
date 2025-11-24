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
@st.cache_data(show_spinner=True)
def load_data():
    loader = BTCDataLoader(kaggle_dataset="mczielinski/bitcoin-historical-data")
    df_hourly, df_daily = loader.load_and_clean(save_dir=None)
    return df_hourly, df_daily

df_hourly, df_daily = load_data()

# -------------------------
# Sidebar navigation
# -------------------------
with st.sidebar:
    selected = option_menu(
        "BTC Dashboard",
        ["Visualizations", "Forecasting"],
        icons=["bar-chart", "graph-up"],
        menu_icon="bitcoin",
        default_index=0,
    )

# -------------------------
# PAGE 1 — VISUALIZATIONS
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
    st.title("📈 Bitcoin Forecasting / Predictive Modeling")
    st.write("Use **daily closing prices** for forecasting models.")
    df = df_daily.copy()

    # ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "Close" not in df.columns:
        st.error("Daily dataset has no 'Close' column — cannot forecast.")
    else:
        st.subheader("Close price (history)")
        st.line_chart(df["Close"], height=300)

        # Basic controls
        st.subheader("Forecast settings")
        days_ahead = int(st.number_input("Forecast days ahead:", 1, 365, 14))
        model_choice = st.selectbox("Model:", ["Naive (last value)", "SMA-extension", "ARIMA (statsmodels)", "Prophet (if installed)"])
        test_size_days = int(st.slider("Holdout size (days) for quick validation", 7, 90, 30))

        # prepare series
        series = df["Close"].asfreq('D').fillna(method="ffill")  # daily freq, forward-fill any missing days

        # train/test split
        train = series.iloc[:-test_size_days]
        test = series.iloc[-test_size_days:]

        st.markdown(f"Training on {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days). "
                    f"Testing on {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days).")

        # utility: plot function (historical + forecast + CI)
        def plot_forecast(train_series, test_series, forecast_index, forecast_mean, conf_int=None, title="Forecast"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_series.index, y=train_series.values, mode='lines', name='Train'))
            fig.add_trace(go.Scatter(x=test_series.index, y=test_series.values, mode='lines', name='Test', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast', line=dict(width=2)))
            if conf_int is not None:
                # conf_int expected as DataFrame with columns ['lower', 'upper'] and same index as forecast_index
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
                fig.add_trace(go.Scatter(
                    x=list(forecast_index) + list(forecast_index[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Confidence Interval'
                ))
            fig.update_layout(height=480, title=title, xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig, use_container_width=True)

        # model implementations
        from sklearn.metrics import mean_absolute_error

        if model_choice == "Naive (last value)":
            last = train.iloc[-1]
            future_index = pd.date_range(series.index[-1], periods=days_ahead + 1, freq='D')[1:]
            forecast_vals = np.repeat(last, len(future_index))
            # quick eval: using test
            pred_test = np.repeat(last, len(test))
            mae = mean_absolute_error(test.values, pred_test)
            st.write(f"MAE on holdout: {mae:,.4f}")
            plot_forecast(train, test, future_index, forecast_vals, conf_int=None, title="Naive Forecast (last value)")

        elif model_choice == "SMA-extension":
            # ensure SMA exists or compute a simple one
            sma_window = st.slider("SMA window (days)", 5, 200, 30)
            if "SMA" not in df.columns:
                sma_series = series.rolling(window=sma_window, min_periods=1).mean()
            else:
                sma_series = df["SMA"].asfreq('D').fillna(method='ffill')
            last_sma = sma_series.iloc[-1]
            future_index = pd.date_range(series.index[-1], periods=days_ahead + 1, freq='D')[1:]
            forecast_vals = np.repeat(last_sma, len(future_index))
            pred_test = np.repeat(last_sma, len(test))
            mae = mean_absolute_error(test.values, pred_test)
            st.write(f"SMA window: {sma_window} days — MAE on holdout: {mae:,.4f}")
            plot_forecast(train, test, future_index, forecast_vals, conf_int=None, title=f"SMA-extension Forecast (window={sma_window})")

        elif model_choice == "ARIMA (statsmodels)":
            # try import statsmodels
            try:
                from statsmodels.tsa.arima.model import ARIMA
                import warnings
                warnings.filterwarnings("ignore")
                st.write("Using statsmodels ARIMA. Tip: adjust p,d,q below if fit fails or residuals look bad.")
                p = int(st.number_input("AR order (p)", 0, 10, 5))
                d = int(st.number_input("Integration order (d)", 0, 2, 1))
                q = int(st.number_input("MA order (q)", 0, 10, 0))

                # Fit on train
                with st.spinner("Fitting ARIMA model..."):
                    model = ARIMA(train.values, order=(p, d, q))
                    model_fit = model.fit()
                st.write(model_fit.summary().tables[1].as_html(), unsafe_allow_html=True)

                # Forecast the test period first (for quick validation), then forecast days_ahead beyond last date
                start = len(train)
                end = len(train) + len(test) - 1
                pred_res = model_fit.get_prediction(start=start, end=end)
                pred_mean_test = pred_res.predicted_mean
                # calculate MAE on holdout
                mae = mean_absolute_error(test.values, pred_mean_test)
                st.write(f"MAE on holdout: {mae:,.4f}")

                # Forecast future
                future_res = model_fit.get_forecast(steps=days_ahead)
                forecast_index = pd.date_range(series.index[-1], periods=days_ahead + 1, freq='D')[1:]
                forecast_mean = future_res.predicted_mean
                conf_int = future_res.conf_int(alpha=0.05)  # dataframe with lower & upper columns

                # conf_int index currently integer; map to forecast_index
                conf_int.index = forecast_index

                plot_forecast(train, test, forecast_index, forecast_mean, conf_int=conf_int, title=f"ARIMA({p},{d},{q}) Forecast")

            except Exception as e:
                st.error("statsmodels ARIMA is not available or model failed to fit. Install statsmodels (`pip install statsmodels`) or choose another model.")
                st.exception(e)

        elif model_choice == "Prophet (if installed)":
            # Prophet has changed package name to 'prophet' (cmdstan/py) — try both imports
            try:
                try:
                    # new package name
                    from prophet import Prophet
                except Exception:
                    # fallback to fbprophet (older)
                    from fbprophet import Prophet

                st.write("Using Prophet. The series will be fit to 'ds' (date) and 'y' (close).")
                # Prepare dataframe for Prophet
                prophet_df = train.reset_index().rename(columns={'index': 'ds', 'Close': 'y'})
                prophet_df = prophet_df[['ds', 'y']]

                m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
                with st.spinner("Fitting Prophet model..."):
                    m.fit(prophet_df)

                # Make future dataframe for validation+forecast
                future_all = m.make_future_dataframe(periods=len(test) + days_ahead, freq='D')
                forecast_all = m.predict(future_all)

                # Extract predicted values for test and future
                pred_test = forecast_all.set_index('ds').loc[test.index]['yhat'].values
                mae = mean_absolute_error(test.values, pred_test)
                st.write(f"MAE on holdout: {mae:,.4f}")

                # Forecast only future days_ahead
                future_df = future_all.tail(days_ahead)
                forecast_future = forecast_all.set_index('ds').loc[future_df['ds']]
                forecast_index = forecast_future.index
                forecast_mean = forecast_future['yhat'].values
                conf_int = forecast_future[['yhat_lower', 'yhat_upper']]
                conf_int.columns = ['lower', 'upper']

                plot_forecast(train, test, forecast_index, forecast_mean, conf_int=conf_int, title="Prophet Forecast")

            except Exception as e:
                st.error("Prophet is not installed or failed to run. Install prophet (`pip install prophet`) or choose ARIMA/SMA.")
                st.exception(e)

        st.info("Note: These are quick interactive models intended for experimentation. For production forecasting consider model validation, hyperparameter tuning, cross-validation, ensembling, and more robust preprocessing (log transforms, outlier handling, exogenous regressors).")
