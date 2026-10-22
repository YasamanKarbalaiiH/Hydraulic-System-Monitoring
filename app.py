import numpy as np
import pandas as pd
import streamlit as st
import logic
from pathlib import Path
import base64

font_path = Path(__file__).parent / "assets" / "Vazirmatn-Regular.woff2"

if font_path.exists():
    encoded_font = base64.b64encode(font_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Vazirmatn';
            src: url(data:font/woff2;base64,{encoded_font}) format('woff2');
            font-weight: 100 900;
            font-style: normal;
        }}

        /* فونت سراسری برنامه */
        html,
        body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stBottom"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stMarkdown"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        [data-testid="stText"],
        [data-testid="stMetric"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        [data-testid="stExpanderDetails"],
        [data-testid="stTabs"],
        [data-testid="stTab"],
        [data-testid="stTable"],
        [data-testid="stDataFrame"],
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-testid="stNumberInput"],
        [data-testid="stTextInput"],
        [data-testid="stTextArea"],
        [data-testid="stDateInput"],
        [data-testid="stTimeInput"],
        [data-testid="stRadio"],
        [data-testid="stCheckbox"],
        [data-testid="stSlider"],
        [data-testid="stFileUploader"],
        [data-testid="stForm"],
        [data-testid="stDialog"],
        [data-testid="stPopover"],
        [data-testid="stStatusWidget"],
        [data-testid="stNotification"],
        button,
        input,
        textarea,
        select,
        option,
        label,
        p,
        a,
        li,
        ul,
        ol,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        small,
        summary{{
            font-family: 'Vazirmatn', sans-serif !important;
        }}

        /* دکمه‌ها */
        button,
        [role="button"] {{
            font-family: 'Vazirmatn', sans-serif !important;
        }}

        /* ورودی‌ها و فهرست‌های انتخاب */
        input,
        textarea,
        select,
        [role="combobox"],
        [role="listbox"],
        [role="option"],
        [role="radio"],
        [role="checkbox"],
        [contenteditable="true"] {{
            font-family: 'Vazirmatn', sans-serif !important;
        }}

        /* جدول‌ها */
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {{
            font-family: 'Vazirmatn', sans-serif !important;
        }}

        /* تیترها */
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 600 !important;
            line-height: 1.7 !important;
        }}

        /* متن عادی */
        p, label, li {{
            font-weight: 400;
            line-height: 1.8;
        }}

        /* کدها با فونت مناسب کدنویسی */
        code, pre {{
            font-family: Consolas, monospace !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



st.set_page_config(
    page_title="Hydraulic System Monitoring",
    page_icon="⚙️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
    }

    section[data-testid="stSidebar"] {
        min-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def print_analysis_report(results):
    """Print the main analysis results to the terminal."""
    print("\n" + "=" * 75)
    print("HYDRAULIC SYSTEM MONITORING - COMPLETE ANALYSIS")
    print("=" * 75)

    print("\n1. DATASET")
    print(f"Number of cycles: {len(results['features'])}")
    print(f"Number of sensors: {len(results['sensor_data'])}")
    print(f"Feature matrix shape: {results['features'].shape}")
    print(f"Profile matrix shape: {results['profile'].shape}")

    print("\nSamples per cycle for each sensor:")
    for sensor, values in results["sensor_data"].items():
        print(f"  {sensor}: {values.shape[1]}")

    print("\n2. TRAIN/TEST SPLIT")
    print(f"Training cycles: {len(results['train_indices'])}")
    print(f"Testing cycles: {len(results['test_indices'])}")

    print("\n3. FEATURE VERIFICATION")
    print("Manual verification for cycle 1:")
    for key, value in results["manual_verification"].items():
        print(f"  {key}: {value}")

    print("\n4. EXTREME FEATURE VALUES")
    print(f"Maximum TS1_mean: {results['max_TS1_mean']}")
    print(f"Maximum VS1_rms: {results['max_VS1_rms']}")

    print("\n5. CORRELATION ANALYSIS")
    print(f"Highly correlated pairs: {len(results['correlations'])}")
    if results["correlations"].empty:
        print("No pairs met the selected correlation threshold.")
    else:
        print(results["correlations"].head(20).to_string(index=False))

    print("\n6. MATRIX RANK AND INDEPENDENT FEATURES")
    independent = results["independent"]
    print(f"Matrix rank: {independent['matrix_rank']}")
    print(
        "Independent features retained: "
        f"{len(independent['independent_indices'])}"
    )
    print(f"Constant columns: {independent['constant_columns']}")
    print(f"Independent feature names: {independent['independent_names']}")

    print("\n7. LARGEST CONSECUTIVE-CYCLE CHANGES")
    print(results["cycle_changes"].to_string(index=False))

    print("\n8. HEALTHY REFERENCE AND THRESHOLDS")
    print(
        "Fully healthy training cycles: "
        f"{results['healthy']['healthy_cycle_count']}"
    )
    print(f"Threshold 95: {results['threshold_95']:.6f}")
    print(f"Threshold 99: {results['threshold_99']:.6f}")
    print(
        "Calibration method: "
        f"{results['thresholds']['calibration_method']}"
    )
    print(
        "Calibration samples: "
        f"{results['thresholds']['calibration_sample_count']}"
    )

    print("\n9. ALERT COUNTS")
    print(
        results["alerts"]["alert"]
        .value_counts()
        .to_string()
    )

    print("\n10. EVALUATION")
    print(results["evaluation"].to_string(index=False))

    evaluation = results["evaluation_summary"]
    print("\nConfusion matrix:")
    for key, value in evaluation["confusion"].items():
        print(f"  {key}: {value}")

    for metric in (
        "detection_rate",
        "false_alarm_rate",
        "precision",
        "accuracy"
    ):
        value = evaluation[metric]
        if np.isfinite(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: undefined")

    print(
        "\nTest cycles with stability issues: "
        f"{evaluation['stability_issue_count']}"
    )

    print("\n11. DISTANCE COMPARISON")
    print(results["distance_comparison"].head(20).to_string(index=False))

    print("\n12. OUTPUT FILES")
    print(f"Output directory: {logic.OUTPUT_DIR.resolve()}")

    if results["healthy"]["healthy_cycle_count"] < 30:
        print(
            "\nWARNING: Few fully healthy training cycles are available. "
            "The thresholds may be unstable and should not be treated "
            "as definitive fault diagnoses."
        )

    print("\n" + "=" * 75)
    print("ANALYSIS REPORT COMPLETE")
    print("=" * 75 + "\n")


def display_metric(value, decimals=4):
    """Format a metric value for Streamlit."""
    if value is None or not np.isfinite(value):
        return "نامشخص"
    return f"{value:.{decimals}f}"


def get_initial_results():
    """Load the dataset for the initial UI before a full analysis."""
    data = logic.load_dataset()

    comparison = logic.prepare_comparison(
    data["features"],
    profile=data["profile"]
)

    feature_df = pd.DataFrame(
        data["features"],
        columns=data["feature_names"]
    )
    feature_df.insert(
        0,
        "cycle_id",
        np.arange(1, len(feature_df) + 1)
    )

    profile_df = pd.DataFrame(
        data["profile"],
        columns=logic.PROFILE_NAMES
    )
    profile_df.insert(
        0,
        "cycle_id",
        np.arange(1, len(profile_df) + 1)
    )

    return {
        **data,
        **comparison,
        "feature_df": feature_df,
        "profile_df": profile_df
    }


st.sidebar.title("Hydraulic Monitor")
st.sidebar.caption("داشبورد پایش تجهیزات هیدرولیک")
st.sidebar.divider()

section = st.sidebar.radio(
    "بخش موردنظر را انتخاب کن",
    [
        "نمای کلی",
        "بررسی چرخه",
        "تحلیل ویژگی‌ها",
        "چرخه‌های مشابه",
        "تحلیل پیشرفته",
        "هشدارها و ارزیابی",
        "ساختار داده‌ها"
    ],
    index=0
)

st.title("پایش هوشمند سیستم هیدرولیک")
st.caption(
    "تحلیل داده‌های سنسورها، بررسی چرخه‌های کاری، "
    "مقایسه وضعیت تجهیزات و شناسایی چرخه‌های نیازمند بررسی"
)

with st.sidebar.expander("تنظیمات تحلیل", expanded=True):
    correlation_threshold = st.slider(
        "حداقل قدر مطلق همبستگی",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
        help="برای نمونه، 0.90 یعنی نمایش زوج‌هایی با |r| حداقل 0.90"
    )

    threshold_95_input = st.number_input(
        "آستانه هشدار اول",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="صفر یعنی استفاده از آستانه پیشنهادی داده‌محور"
    )

    threshold_99_input = st.number_input(
        "آستانه هشدار دوم",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="صفر یعنی استفاده از آستانه پیشنهادی داده‌محور"
    )

    run_button = st.button(
        "اجرای کامل تحلیل",
        type="primary",
        width="stretch"
    )


if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None


if run_button:
    try:
        with st.spinner("در حال اجرای تمام مراحل تحلیل..."):
            results = logic.run_analysis(
                threshold_95=(
                    None
                    if threshold_95_input == 0
                    else threshold_95_input
                ),
                threshold_99=(
                    None
                    if threshold_99_input == 0
                    else threshold_99_input
                ),
                correlation_threshold=correlation_threshold
            )

        st.session_state["analysis_results"] = results

        print_analysis_report(results)

        st.sidebar.success("تحلیل کامل شد؛ گزارش در ترمینال چاپ شد")
        st.success(
            "تحلیل با موفقیت انجام شد. نتایج در داشبورد و ترمینال "
            "قابل مشاهده‌اند"
        )

    except Exception as error:
        st.error(
            f"اجرای تحلیل ناموفق بود: {type(error).__name__}: {error}"
        )
        print(
            f"\nANALYSIS ERROR: {type(error).__name__}: {error}"
        )


if st.session_state["analysis_results"] is None:
    try:
        with st.spinner("در حال بارگذاری داده‌ها برای نمایش اولیه..."):
            results = get_initial_results()

    except Exception as error:
        st.error(
            f"خطا در بارگذاری داده‌ها: {type(error).__name__}: {error}"
        )
        st.stop()
else:
    results = st.session_state["analysis_results"]


sensor_data_dict = results["sensor_data"]
features = results["features"]
feature_names = results["feature_names"]
profile = results["profile"]
features_df = results["feature_df"]
profile_df = results["profile_df"]
train_indices = results["train_indices"]
test_indices = results["test_indices"]
X_train_scaled = results["X_train_scaled"]
X_test_scaled = results["X_test_scaled"]


st.sidebar.divider()
st.sidebar.success("داده‌ها بارگذاری شده‌اند")
st.sidebar.caption(f"تعداد چرخه‌ها: {len(features):,}")
st.sidebar.caption(f"تعداد سنسورها: {len(sensor_data_dict)}")
st.sidebar.caption(f"تعداد ویژگی‌ها: {len(feature_names)}")
st.sidebar.caption("ورودی‌ها فقط از رابط کاربری دریافت می‌شوند")


if section == "نمای کلی":
    st.subheader("نمای کلی سیستم")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تعداد چرخه‌ها", f"{len(features):,}")
    c2.metric("تعداد سنسورها", len(sensor_data_dict))
    c3.metric("تعداد ویژگی‌ها", len(feature_names))
    c4.metric("متغیرهای وضعیت", len(logic.PROFILE_NAMES))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### داده‌های ورودی")
        st.write(
            "۱۷ فایل سنسور و یک فایل وضعیت تجهیزات برای ۲۲۰۵ چرخه کاری"
        )
        sensor_samples_df = pd.DataFrame([
            {
                "سنسور": name,
                "تعداد چرخه‌ها": values.shape[0],
                "نمونه در هر چرخه": values.shape[1]
            }
            for name, values in sensor_data_dict.items()
        ])
        st.dataframe(
            sensor_samples_df,
            hide_index=True,
            width="stretch"
        )

        st.write(
            "از هر سنسور پنج ویژگی محاسبه می‌شود: میانگین، حداقل، "
            "حداکثر، انحراف معیار و RMS"
        )

    with right:
        st.markdown("#### تحلیل‌ها")
        st.write(
            "استانداردسازی با استفاده از مجموعه آموزش، فاصله اقلیدسی، "
            "شباهت کسینوسی، همبستگی، رتبه ماتریس، تغییرات چرخه‌ها و هشداردهی"
        )
        st.info(
            "برای اجرای تمام مراحل و چاپ گزارش در ترمینال، "
            "دکمه «اجرای کامل تحلیل» را در نوار کناری بزن"
        )

    st.divider()
    st.subheader("پیش‌نمایش ویژگی‌های استخراج‌شده")
    st.dataframe(
        features_df.head(10),
        hide_index=True,
        width="stretch"
    )

    if "thresholds" in results:
        st.subheader("خلاصه آخرین تحلیل")

        c1, c2, c3 = st.columns(3)
        c1.metric("آستانه اول", f"{results['threshold_95']:.4f}")
        c2.metric("آستانه دوم", f"{results['threshold_99']:.4f}")
        c3.metric(
            "چرخه سالم در آموزش",
            results["healthy"]["healthy_cycle_count"]
        )

        st.markdown("#### ارزیابی عملکرد")
        evaluation = results["evaluation_summary"]

        a, b, c, d = st.columns(4)
        a.metric("TP", evaluation["confusion"]["TP"])
        b.metric("FP", evaluation["confusion"]["FP"])
        c.metric("FN", evaluation["confusion"]["FN"])
        d.metric("TN", evaluation["confusion"]["TN"])


elif section == "بررسی چرخه":
    st.subheader("بررسی اطلاعات یک چرخه")

    cycle_number = st.number_input(
        "شماره چرخه را انتخاب کن",
        min_value=1,
        max_value=len(features),
        value=1,
        step=1
    )

    cycle_index = int(cycle_number) - 1

    st.markdown(
        f"#### پنج نمونه اول هر سنسور — چرخه {int(cycle_number)}"
    )

    raw_data = []

    for sensor_name in logic.SENSOR_NAMES:
        readings = sensor_data_dict[sensor_name][cycle_index, :5]

        row = {
            "سنسور": sensor_name
        }

        for i, reading in enumerate(readings, start=1):
            row[f"نمونه {i}"] = reading

        raw_data.append(row)

    st.dataframe(
        pd.DataFrame(raw_data),
        hide_index=True,
        width="stretch"
    )

    st.markdown("#### ویژگی‌های آماری")

    cycle_features_df = pd.DataFrame({
        "ویژگی": feature_names,
        "مقدار": features[cycle_index]
    })

    st.dataframe(
        cycle_features_df,
        hide_index=True,
        width="stretch"
    )

    st.markdown("#### وضعیت ثبت‌شده تجهیزات")

    status_cols = st.columns(5)

    for column, name, value in zip(
        status_cols,
        logic.PROFILE_NAMES,
        profile[cycle_index]
    ):
        column.metric(name, f"{value:g}")

    equipment_fault = not np.all(
        np.isclose(
            profile[cycle_index, :4],
            logic.IDEAL_PROFILE[:4]
        )
    )

    stability_issue = not np.isclose(
        profile[cycle_index, 4],
        logic.IDEAL_PROFILE[4]
    )

    if equipment_fault:
        st.warning(
            "طبق پروفایل ثبت‌شده، حداقل یکی از چهار متغیر اصلی "
            "تجهیز با مقدار ایده‌آل تفاوت دارد"
        )
    else:
        st.success(
            "چهار متغیر اصلی تجهیز با مقادیر ایده‌آل برابرند"
        )

    if stability_issue:
        st.warning(
            "متغیر Stability با مقدار ایده‌آل تفاوت دارد"
        )
    else:
        st.info(
            "متغیر Stability با مقدار ایده‌آل برابر است"
        )

    if "alerts" in results:
        matching_alert = results["alerts"].loc[
            results["alerts"]["cycle_id"] == int(cycle_number)
        ]

        if not matching_alert.empty:
            st.markdown("#### هشدار محاسبه‌شده")
            st.dataframe(
                matching_alert,
                hide_index=True,
                width="stretch"
            )


elif section == "تحلیل ویژگی‌ها":
    st.subheader("تحلیل آماری ویژگی‌ها")

    default_feature_index = (
        feature_names.index("TS1_mean")
        if "TS1_mean" in feature_names
        else 0
    )

    selected_feature = st.selectbox(
        "ویژگی مورد بررسی را انتخاب کن",
        options=feature_names,
        index=default_feature_index
    )

    feature_index = feature_names.index(selected_feature)
    values = features[:, feature_index]

    max_index = int(np.argmax(values))
    min_index = int(np.argmin(values))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("میانگین", f"{np.mean(values):.4f}")
    c2.metric("کمترین مقدار", f"{np.min(values):.4f}")
    c3.metric("بیشترین مقدار", f"{np.max(values):.4f}")
    c4.metric("انحراف معیار", f"{np.std(values):.4f}")

    st.markdown("#### تغییرات ویژگی در چرخه‌ها")

    chart_df = pd.DataFrame({
        "شماره چرخه": np.arange(1, len(values) + 1),
        selected_feature: values
    }).set_index("شماره چرخه")

    st.line_chart(chart_df)

    a, b = st.columns(2)
    a.info(
        f"بیشترین مقدار در چرخه {max_index + 1}: "
        f"{values[max_index]:.4f}"
    )
    b.info(
        f"کمترین مقدار در چرخه {min_index + 1}: "
        f"{values[min_index]:.4f}"
    )

    st.markdown("#### توزیع مقادیر")
    st.bar_chart(pd.DataFrame({selected_feature: values}))

    if "max_TS1_mean" in results:
        st.divider()

        st.write(
            f"بیشترین TS1_mean: "
            f"{results['max_TS1_mean']['value']:.4f} "
            f"در چرخه {results['max_TS1_mean']['cycle_id']}."
        )

        st.write(
            f"بیشترین VS1_rms: "
            f"{results['max_VS1_rms']['value']:.4f} "
            f"در چرخه {results['max_VS1_rms']['cycle_id']}."
        )

    if "standardization_params" in results:
      st.divider()
      st.markdown("#### پارامترهای استانداردسازی ویژگی انتخاب‌شده")

      params = results["standardization_params"].copy()

      params["feature_name"] = feature_names

      feature_param = params.loc[
        params["feature_name"] == selected_feature
    ]

      if not feature_param.empty:
        st.dataframe(
            feature_param,
            hide_index=True,
            width="stretch"
        )
      else:
        st.warning("پارامترهای استانداردسازی این ویژگی پیدا نشدند")

elif section == "چرخه‌های مشابه":
    st.subheader("مقایسه چرخه‌های کاری")

    test_cycle_ids = sorted((test_indices + 1).tolist())

    query_cycle_id = st.selectbox(
        "چرخه مورد مقایسه را از مجموعه آزمون انتخاب کن",
        options=test_cycle_ids
    )

    if st.button("مقایسه چرخه", type="primary"):
        try:
            similar = logic.find_similar_cycles(
                query_cycle_id,
                train_indices,
                test_indices,
                X_train_scaled,
                X_test_scaled
            )

            st.markdown("#### پنج چرخه نزدیک‌تر با فاصله اقلیدسی")
            st.dataframe(
                similar["distance_df"],
                hide_index=True,
                width="stretch"
            )

            st.markdown("#### پنج چرخه مشابه‌تر با شباهت کسینوسی")
            st.dataframe(
                similar["cosine_df"],
                hide_index=True,
                width="stretch"
            )

            nearest_pos = int(similar["nearest_positions"][0])
            nearest_cycle_id = int(train_indices[nearest_pos] + 1)

            st.info(
                f"چرخه {query_cycle_id} از نظر فاصله اقلیدسی "
                f"به چرخه {nearest_cycle_id} نزدیک‌تر است"
                "شباهت به معنی سالم‌بودن قطعی نیست"
            )

            differences = np.abs(
                similar["query_scaled"] - X_train_scaled[nearest_pos]
            )

            top = np.argsort(differences)[::-1][:5]

            diff_df = pd.DataFrame({
                "ویژگی": [feature_names[i] for i in top],
                "اختلاف مطلق استانداردشده": differences[top]
            })

            st.markdown("#### پنج ویژگی با بیشترین اختلاف")
            st.dataframe(
                diff_df,
                hide_index=True,
                width="stretch"
            )

            print("\n--- Similar-cycle analysis ---")
            print(f"Query cycle: {query_cycle_id}")
            print("Nearest by Euclidean distance:")
            print(similar["distance_df"].to_string(index=False))
            print("Most similar by cosine similarity:")
            print(similar["cosine_df"].to_string(index=False))
            print("Largest standardized feature differences:")
            print(diff_df.to_string(index=False))

        except Exception as error:
            st.error(f"خطا در مقایسه چرخه‌ها: {error}")
            print(f"SIMILAR-CYCLE ERROR: {error}")


elif section == "تحلیل پیشرفته":
    st.subheader("تحلیل‌های تکمیلی")

    if "correlations" not in results:
        st.info(
            "ابتدا از نوار کناری «اجرای کامل تحلیل» را اجرا کن "
            "تا نتایج تکمیلی محاسبه شوند"
        )
    else:
        st.markdown("#### زوج ویژگی‌های با همبستگی بالا")

        st.caption(
            "همبستگی از مجموعه آموزش محاسبه شده است "
            "ستون‌های ثابت در رتبه‌بندی زوج‌ها لحاظ نمی‌شوند"
        )

        st.write(
            f"تعداد زوج‌ها با |r| ≥ {correlation_threshold:.2f}: "
            f"{len(results['correlations'])}"
        )

        st.dataframe(
            results["correlations"].head(100),
            hide_index=True,
            width="stretch"
        )

        st.divider()
        st.markdown("#### رتبه ماتریس و ویژگی‌های مستقل")

        independent = results["independent"]

        a, b, c = st.columns(3)
        a.metric("رتبه ماتریس آموزش", independent["matrix_rank"])
        b.metric(
            "ویژگی‌های مستقل نگه‌داشته‌شده",
            len(independent["independent_indices"])
        )
        c.metric(
            "ستون‌های ثابت",
            len(independent["constant_columns"])
        )

        st.write("ویژگی‌های ثابت:")
        st.write(
            ", ".join(independent["constant_columns"])
            if independent["constant_columns"]
            else "هیچ‌کدام"
        )

        st.write("ویژگی‌های انتخاب‌شده به‌عنوان مستقل:")
        st.dataframe(
            pd.DataFrame({
                "ویژگی": independent["independent_names"]
            }),
            hide_index=True,
            width="stretch"
        )

        st.divider()
        st.markdown("#### ده تغییر بزرگ بین چرخه‌های متوالی")

        st.dataframe(
            results["cycle_changes"],
            hide_index=True,
            width="stretch"
        )

        st.divider()
        st.markdown(
            "#### مقایسه فاصله با همه ویژگی‌ها و ویژگی‌های مستقل"
        )

        st.caption(
            "این جدول برای اولین چرخه مجموعه آزمون محاسبه شده است"
        )

        st.dataframe(
            results["distance_comparison"].head(20),
            hide_index=True,
            width="stretch"
        )

        st.divider()
        st.markdown("#### پارامترهای استانداردسازی")

        st.dataframe(
            results["standardization_params"],
            hide_index=True,
            width="stretch"
        )


elif section == "هشدارها و ارزیابی":
    st.subheader("هشدارهای وضعیت و ارزیابی")

    if "alerts" not in results:
        st.info(
            "ابتدا از نوار کناری «اجرای کامل تحلیل» را اجرا کن"
        )
    else:
        thresholds = results["thresholds"]

        a, b, c = st.columns(3)
        a.metric("آستانه Normal", f"{results['threshold_95']:.4f}")
        b.metric("آستانه Suspicious", f"{results['threshold_99']:.4f}")
        c.metric(
            "چرخه سالم آموزش",
            thresholds["healthy_cycle_count"]
        )

        st.caption(
            f"روش کالیبراسیون: {thresholds['calibration_method']}. "
            f"تعداد نمونه‌های کالیبراسیون: "
            f"{thresholds['calibration_sample_count']}."
        )

        if thresholds["healthy_cycle_count"] < 30:
            st.warning(
                "تعداد چرخه‌های کاملاً سالم در مجموعه آموزش کم است؛ "
                "آستانه‌ها ممکن است ناپایدار باشند. "
                "این خروجی ابزار غربالگری است، نه تشخیص قطعی خرابی"
            )

        alert_counts = (
            results["alerts"]["alert"]
            .value_counts()
            .rename_axis("وضعیت هشدار")
            .reset_index(name="تعداد")
        )

        st.markdown("#### توزیع هشدارها در مجموعه آزمون")

        st.dataframe(
            alert_counts,
            hide_index=True,
            width="stretch"
        )

        st.bar_chart(
            alert_counts.set_index("وضعیت هشدار")
        )

        st.markdown("#### جدول هشدارها")

        filter_alert = st.selectbox(
            "فیلتر وضعیت",
            [
                "همه",
                "Normal",
                "Suspicious",
                "Needs inspection"
            ]
        )

        alert_df = results["alerts"]

        if filter_alert != "همه":
            alert_df = alert_df[
                alert_df["alert"] == filter_alert
            ]

        st.dataframe(
            alert_df,
            hide_index=True,
            width="stretch"
        )

        st.download_button(
            "دانلود جدول هشدارها",
            data=results["alerts"].to_csv(index=False).encode("utf-8-sig"),
            file_name="alerts.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("#### ارزیابی با پروفایل واقعی")

        evaluation = results["evaluation_summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TP", evaluation["confusion"]["TP"])
        c2.metric("FP", evaluation["confusion"]["FP"])
        c3.metric("FN", evaluation["confusion"]["FN"])
        c4.metric("TN", evaluation["confusion"]["TN"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Detection rate",
            display_metric(evaluation["detection_rate"] * 100, 2)
            + "%" if np.isfinite(evaluation["detection_rate"])
            else "نامشخص"
        )

        c2.metric(
            "False alarm rate",
            display_metric(evaluation["false_alarm_rate"] * 100, 2)
            + "%" if np.isfinite(evaluation["false_alarm_rate"])
            else "نامشخص"
        )

        c3.metric(
            "Precision",
            display_metric(evaluation["precision"] * 100, 2)
            + "%" if np.isfinite(evaluation["precision"])
            else "نامشخص"
        )

        c4.metric(
            "Accuracy",
            display_metric(evaluation["accuracy"] * 100, 2)
            + "%" if np.isfinite(evaluation["accuracy"])
            else "نامشخص"
        )

        st.dataframe(
            results["evaluation"],
            hide_index=True,
            width="stretch"
        )

        st.download_button(
            "دانلود نتایج ارزیابی",
            data=results["evaluation"].to_csv(index=False).encode("utf-8-sig"),
            file_name="evaluation.csv",
            mime="text/csv"
        )

        st.caption(
            "خرابی تجهیز با تفاوت هر یک از چهار متغیر اصلی از مقدار "
            "ایده‌آل تعریف شده است. Stability جداگانه گزارش می‌شود"
        )


elif section == "ساختار داده‌ها":
    st.subheader("ساختار و اعتبارسنجی داده‌ها")

    structure_df = pd.DataFrame({
        "بخش داده": [
            "داده خام هر سنسور",
            "ماتریس ویژگی‌ها",
            "فایل وضعیت تجهیزات",
            "مجموعه آموزش",
            "مجموعه آزمون"
        ],
        "ابعاد": [
            str(sensor_data_dict[logic.SENSOR_NAMES[0]].shape),
            str(features.shape),
            str(profile.shape),
            str(X_train_scaled.shape),
            str(X_test_scaled.shape)
        ],
        "توضیح": [
            "نمونه‌های خام یک سنسور در هر چرخه",
            "۸۵ ویژگی از ۱۷ سنسور",
            "چهار متغیر اصلی تجهیز و متغیر پایداری",
            "۸۰ درصد چرخه‌ها؛ استانداردسازی بر مبنای آموزش",
            "۲۰ درصد چرخه‌ها؛ با پارامترهای آموزش استاندارد شده‌اند"
        ]
    })

    st.dataframe(
        structure_df,
        hide_index=True,
        width="stretch"
    )

    st.divider()
    st.markdown("#### تعداد نمونه‌های فایل‌های سنسور")

    sensor_shapes = pd.DataFrame([
        {
            "سنسور": name,
            "تعداد چرخه‌ها": values.shape[0],
            "نمونه در هر چرخه": values.shape[1],
            "داده معتبر": bool(np.isfinite(values).all())
        }
        for name, values in sensor_data_dict.items()
    ])

    st.dataframe(
        sensor_shapes,
        hide_index=True,
        width="stretch"
    )

    st.divider()
    st.markdown("#### توزیع مقادیر پروفایل")

    profile_summary = []

    for index, name in enumerate(logic.PROFILE_NAMES):
        values, counts = np.unique(
            profile[:, index],
            return_counts=True
        )

        for value, count in zip(values, counts):
            profile_summary.append({
                "متغیر": name,
                "مقدار": value,
                "تعداد چرخه‌ها": int(count),
                "درصد": round(
                    count / len(profile) * 100,
                    2
                )
            })

    st.dataframe(
        pd.DataFrame(profile_summary),
        hide_index=True,
        width="stretch"
    )

    st.divider()
    st.markdown("#### بررسی مقادیر نامعتبر")

    invalid_summary = pd.DataFrame([
        {
            "بخش": name,
            "شکل داده": str(values.shape),
            "تعداد NaN یا Inf": int(
                (~np.isfinite(values)).sum()
            )
        }
        for name, values in sensor_data_dict.items()
    ])

    invalid_summary.loc[len(invalid_summary)] = {
        "بخش": "ماتریس ویژگی‌ها",
        "شکل داده": str(features.shape),
        "تعداد NaN یا Inf": int(
            (~np.isfinite(features)).sum()
        )
    }

    invalid_summary.loc[len(invalid_summary)] = {
        "بخش": "Profile",
        "شکل داده": str(profile.shape),
        "تعداد NaN یا Inf": int(
            (~np.isfinite(profile)).sum()
        )
    }

    st.dataframe(
        invalid_summary,
        hide_index=True,
        width="stretch"
    )

    st.divider()
    st.markdown("#### فایل‌های خروجی")

    if logic.OUTPUT_DIR.exists():
        output_files = sorted(
            path for path in logic.OUTPUT_DIR.iterdir()
            if path.is_file()
        )

        if output_files:
            st.dataframe(
                pd.DataFrame({
                    "نام فایل": [path.name for path in output_files],
                    "مسیر": [str(path) for path in output_files],
                    "اندازه (بایت)": [
                        path.stat().st_size for path in output_files
                    ]
                }),
                hide_index=True,
                width="stretch"
            )
        else:
            st.info("هنوز فایل خروجی‌ای تولید نشده است")
    else:
        st.info(
            "پوشه outputs هنوز ایجاد نشده است "
            "ابتدا تحلیل کامل را اجرا کن"
        )


st.sidebar.divider()
st.sidebar.caption(
    "پس از اجرای کامل تحلیل، گزارش نتایج در ترمینال چاپ می‌شود"
)
