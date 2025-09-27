import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

# ----- Background image -----
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://img.freepik.com/premium-photo/planet-green-background-background-ecology-theme-with-space-text-green-grass-clean-planet_339391-94.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 1rem 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ----- Load Data -----
emissions_df = pd.read_csv("emissions_low_granularity.csv")
emissions_df['parent_entity'] = emissions_df['parent_entity'].str.lower()

# Load trained regression model
model = joblib.load("linear_carbon_model.pkl")

# ----- Sidebar Menu -----
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Overview", "Forecast & Predictions", "What-if Simulator", "NetZero 2050", "Rankings"]
)

# ----- Company Selector -----
unique_companies = sorted(emissions_df['parent_entity'].unique())
company_name = st.sidebar.selectbox("🏢 Select a Company", unique_companies)

if company_name:
    company_data = emissions_df[emissions_df['parent_entity'] == company_name.strip().lower()]

    if company_data.empty:
        st.error("Company not found.")
    else:
        company_data = company_data.sort_values(by='year')

        # Latest year info
        latest_row = company_data.iloc[-1]
        latest_year = latest_row['year']
        latest_emissions = latest_row['total_emissions_MtCO2e']
        input_df = pd.DataFrame({'total_emissions_MtCO2e': [latest_emissions]})
        predicted_credits = model.predict(input_df)[0]

        # ----- OVERVIEW -----
        if menu == "Overview":
            st.title("🌍 Carbon Emissions Dashboard")
            st.info("ℹ️ **Emissions (MtCO₂e)** means 'Million tons of CO₂ equivalent'. "
                    "It combines all greenhouse gases (CO₂, methane, etc.) into one number.")

            st.subheader(f"📊 Emissions History: {company_name.title()}")
            st.line_chart(company_data.set_index('year')['total_emissions_MtCO2e'])

            st.success(
                f"✅ In {latest_year}, {company_name.title()} emitted **{latest_emissions:.2f} MtCO₂e** "
                f"and would need about **{int(predicted_credits):,} Carbon Credits**."
            )
            st.info("💡 Carbon Credits are certificates companies buy to balance out emissions they cannot reduce directly.")

        # ----- FORECAST -----
        elif menu == "Forecast & Predictions":
            st.title("🔮 AI Forecast (Next 5 Years)")
            st.info("📌 Forecasts are based on past data. They give a **best estimate** of future emissions, "
                    "but real numbers can be higher or lower.")

            prophet_df = company_data.rename(columns={"year": "ds", "total_emissions_MtCO2e": "y"})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
            model_prophet = Prophet(yearly_seasonality=False, daily_seasonality=False)
            model_prophet.fit(prophet_df)

            future = model_prophet.make_future_dataframe(periods=5, freq='Y')
            forecast = model_prophet.predict(future)
            forecast['carbon_credits'] = model.predict(
                forecast[['yhat']].rename(columns={'yhat': 'total_emissions_MtCO2e'})
            )

            fig, ax = plt.subplots()
            ax.plot(forecast['ds'], forecast['yhat'], label="Predicted Emissions", color="green")
            ax.set_ylabel("Emissions (MtCO₂e)")
            ax.legend()
            st.pyplot(fig)

        # ----- WHAT-IF -----
        elif menu == "What-if Simulator":
            st.title("🛠 What-if Simulator")
            st.warning("ℹ️ This tool lets you **simulate scenarios**. Example: "
                       "What happens if emissions drop by 10% next year?")

            reduction = st.slider("Reduce emissions by (%)", 0, 50, 10)
            adjusted_emissions = latest_emissions * (1 - reduction / 100)
            adjusted_input = pd.DataFrame({'total_emissions_MtCO2e': [adjusted_emissions]})
            adjusted_credits = model.predict(adjusted_input)[0]

            st.success(
                f"➡️ If {company_name.title()} reduced emissions by **{reduction}%** in {latest_year}, "
                f"carbon credits needed would drop to **{int(adjusted_credits):,}** "
                f"(saving {int(predicted_credits - adjusted_credits):,})."
            )

        # ----- NETZERO -----
        elif menu == "NetZero 2050":
            st.title("🌐 NetZero 2050 Alignment")
            st.info("🌱 NetZero 2050 means a company’s emissions will fall close to **zero by the year 2050**. "
                    "Any remaining emissions should be balanced with carbon removal.")

            # Train Prophet model for long-term forecast
            prophet_df = company_data.rename(columns={"year": "ds", "total_emissions_MtCO2e": "y"})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
            model_prophet = Prophet(yearly_seasonality=False, daily_seasonality=False)
            model_prophet.fit(prophet_df)

            netzero_years = list(range(latest_year, 2051))
            netzero_path = pd.Series(
                [latest_emissions * (1 - (y - latest_year) / (2050 - latest_year)) for y in netzero_years],
                index=netzero_years
            )

            future_2050 = model_prophet.make_future_dataframe(periods=(2050 - latest_year), freq='Y')
            forecast_2050 = model_prophet.predict(future_2050)

            fig4, ax = plt.subplots()
            ax.plot(forecast_2050['ds'].dt.year, forecast_2050['yhat'], label="Forecasted Emissions", color="green")
            ax.plot(netzero_path.index, netzero_path.values, label="NetZero 2050 Pathway", color="red", linestyle="--")
            ax.set_ylabel("Emissions (MtCO₂e)")
            ax.legend()
            st.pyplot(fig4)

            forecast_2050_last = forecast_2050.iloc[-1]['yhat']
            if forecast_2050_last <= 0.05 * latest_emissions:
                st.success(f"✅ {company_name.title()} is on track for NetZero 2050.")
            else:
                st.error(f"❌ {company_name.title()} is NOT aligned with NetZero 2050. "
                         f"Projected 2050 emissions: {forecast_2050_last:.2f} MtCO₂e.")

        # ----- RANKINGS -----
        elif menu == "Rankings":
            st.title("🏆 NetZero Leaders & Laggards")
            st.info("This shows which companies are reducing fastest (leaders) "
                    "and which ones are slowest (laggards).")

            rankings = []
            for comp in emissions_df['parent_entity'].unique():
                comp_data = emissions_df[emissions_df['parent_entity'] == comp].sort_values('year')
                if len(comp_data) > 3:
                    latest_year = comp_data.iloc[-1]['year']
                    latest_emissions = comp_data.iloc[-1]['total_emissions_MtCO2e']

                    prophet_df = comp_data.rename(columns={"year": "ds", "total_emissions_MtCO2e": "y"})
                    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
                    m = Prophet(yearly_seasonality=False, daily_seasonality=False)
                    m.fit(prophet_df)

                    f2050 = m.make_future_dataframe(periods=(2050 - latest_year), freq='Y')
                    pred2050 = m.predict(f2050)
                    final_val = pred2050.iloc[-1]['yhat']

                    progress = (1 - final_val / latest_emissions) * 100
                    rankings.append((comp.title(), progress))

            rank_df = pd.DataFrame(rankings, columns=["Company", "Reduction by 2050 (%)"])
            leaders = rank_df.sort_values("Reduction by 2050 (%)", ascending=False).head(10)
            laggards = rank_df.sort_values("Reduction by 2050 (%)", ascending=True).head(10)

            st.write("### 🌟 Top 10 NetZero Leaders")
            st.table(leaders)

            st.write("### ⚠️ Top 10 NetZero Laggards")
            st.table(laggards)
