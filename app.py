import streamlit as st

from src.data_loading import read_dataset
from src.profiling import profile_dataframe


def main() -> None:
    st.set_page_config(page_title="Medical Dataset Explorer", layout="wide")

    st.title("Medical Dataset Explorer")

    upload_tab = st.tabs(["Upload"])[0]

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a dataset",
            type=["csv", "tsv", "xlsx"],
            help="Supported formats: CSV, TSV, XLSX",
        )
        st.caption("Supported formats: CSV, TSV, XLSX")

        if uploaded_file is None:
            st.info("Upload a CSV, TSV, or Excel file to begin.")
            return

        try:
            df = read_dataset(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success("Dataset loaded successfully:")
        st.markdown(
            "\n".join(
                [
                    f"- File: {uploaded_file.name}",
                    f"- Rows: {len(df)}",
                    f"- Columns: {len(df.columns)}",
                ]
            )
        )

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Column profile")
        profile = profile_dataframe(df)
        st.dataframe(profile, use_container_width=True)


if __name__ == "__main__":
    main()
