else:
    st.markdown(
        """
        <div style="font-size:0.78rem;color:#666;margin:0 0 10px 0">
            Upload file CSV dengan minimal kolom:
            <code>Kecamatan</code>, <code>Tanggal Servis</code>, dan
            <code>Permintaan Servis</code>.
            <br>
            Nama alternatif seperti <code>Kecamatan Bengkel</code>,
            <code>Jumlah Servis</code>, atau <code>Tanggal</code> juga akan diterima.
        </div>
        """,
        unsafe_allow_html=True
    )

    # CSS khusus uploader, tanpa menyembunyikan elemen internal Streamlit
    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] {
        margin-top: 8px;
        margin-bottom: 18px;
        max-width: 720px;
    }

    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #dddddd !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        padding: 18px !important;
        min-height: 120px !important;
    }

    div[data-testid="stFileUploader"] button {
        border-radius: 8px !important;
        font-size: 0.78rem !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #888 !important;
        font-size: 0.72rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV data aktual",
        type=["csv"],
        label_visibility="visible"
    )

    def normalize_col(col):
        return (
            str(col)
            .replace("\ufeff", "")
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

    def normalize_district(x):
        x = str(x).strip()

        # Kalau sudah format DISTRICT_A, biarkan
        if x.upper().startswith("DISTRICT_"):
            return x.upper()

        # Kalau format District A, ubah menjadi DISTRICT_A
        if x.lower().startswith("district "):
            return "DISTRICT_" + x.split()[-1].upper()

        return x

    if uploaded is not None:
        try:
            # sep=None supaya bisa baca CSV koma atau titik koma
            df_up = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")

            # Hilangkan spasi dan BOM pada nama kolom
            original_cols = list(df_up.columns)
            col_map = {normalize_col(c): c for c in df_up.columns}

            aliases = {
                "Kecamatan": [
                    "kecamatan",
                    "kecamatan bengkel",
                    "district",
                    "nama kecamatan"
                ],
                "Tanggal Servis": [
                    "tanggal servis",
                    "tanggal",
                    "date",
                    "tanggal service",
                    "minggu"
                ],
                "Permintaan Servis": [
                    "permintaan servis",
                    "jumlah servis",
                    "aktual",
                    "actual",
                    "demand",
                    "service demand"
                ]
            }

            rename_dict = {}

            for target_col, possible_names in aliases.items():
                found = None
                for name in possible_names:
                    if name in col_map:
                        found = col_map[name]
                        break

                if found is not None:
                    rename_dict[found] = target_col

            df_up = df_up.rename(columns=rename_dict)

            required = ["Kecamatan", "Tanggal Servis", "Permintaan Servis"]
            missing = [c for c in required if c not in df_up.columns]

            if missing:
                st.error(
                    f"Kolom tidak ditemukan: {missing}. "
                    f"Kolom yang terbaca dari file: {original_cols}"
                )
                st.stop()

            df_up = df_up[required].copy()

            df_up["Kecamatan"] = df_up["Kecamatan"].apply(normalize_district)
            df_up["Tanggal Servis"] = pd.to_datetime(
                df_up["Tanggal Servis"],
                errors="coerce"
            )
            df_up["Permintaan Servis"] = pd.to_numeric(
                df_up["Permintaan Servis"],
                errors="coerce"
            )

            df_up = df_up.dropna(
                subset=["Kecamatan", "Tanggal Servis", "Permintaan Servis"]
            )

            if df_up.empty:
                st.warning("File berhasil dibaca, tetapi tidak ada baris valid setelah pembersihan data.")
                st.stop()

            st.success(f"File berhasil dibaca: {len(df_up)} baris data valid.")

            rows_up = []
            merged_up = {}

            for kec in TOP5:
                act = df_up[
                    df_up["Kecamatan"] == kec
                ][["Tanggal Servis", "Permintaan Servis"]].copy()

                if act.empty:
                    continue

                merged = pd.merge(
                    fcr[kec][["Tanggal Servis", "forecast"]],
                    act,
                    on="Tanggal Servis",
                    how="inner"
                )

                merged = merged[merged["Permintaan Servis"] > 0]

                if merged.empty:
                    continue

                fc_v = merged["forecast"].values
                ac_v = merged["Permintaan Servis"].values

                rows_up.append({
                    "Kecamatan": kec,
                    "District": dl(kec),
                    "MAPE (%)": round(np.mean(np.abs(fc_v - ac_v) / ac_v) * 100, 1),
                    "RMSE": int(np.sqrt(np.mean((fc_v - ac_v) ** 2))),
                    "MAE": int(np.mean(np.abs(fc_v - ac_v))),
                    "Bias": int(np.mean(fc_v - ac_v)),
                    "Minggu": len(merged),
                })

                merged_up[kec] = merged.rename(
                    columns={"Permintaan Servis": "aktual"}
                )

            if rows_up:
                tgl_min = df_up["Tanggal Servis"].min().strftime("%d %b %Y")
                tgl_max = df_up["Tanggal Servis"].max().strftime("%d %b %Y")
                render_perbandingan(
                    pd.DataFrame(rows_up),
                    merged_up,
                    f"{tgl_min} – {tgl_max}"
                )
            else:
                st.warning(
                    "File terbaca, tetapi tidak ada data yang cocok dengan forecast. "
                    "Pastikan nama district sesuai, misalnya DISTRICT_A atau District A, "
                    "dan tanggalnya sama dengan periode forecast."
                )

                st.dataframe(
                    df_up.head(10),
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
