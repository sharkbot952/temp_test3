# app_calendar_streamlit_test.py
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from matplotlib import rcParams

rcParams['font.family'] = 'Meiryo'

# =========================================
# CSS（上部バー・フッター非表示・フォント小さめ）
# =========================================
st.markdown("""
<style>
    footer {visibility: hidden;}
    .block-container {padding: 0.2rem 0.5rem 0.2rem 0.5rem !important;}
    div[data-baseweb="select"] * {font-size: 0.8rem !important; line-height: 1.2rem !important;margin-bottom: 0.1rem !important}
    div[data-baseweb="radio"] * {font-size: 0.6rem !important; line-height: 0.7rem !important; margin:0 !important; padding:0 !important;}
</style>
""", unsafe_allow_html=True)

# =========================================
# データ読み込み
# =========================================
def load_uv_ts(folder):
    files_uv = glob.glob(os.path.join(folder, "u_v_*.csv"))
    files_ts = glob.glob(os.path.join(folder, "t_s_*.csv"))

    def extract_depth(fname):
        m = re.search(r"Lv(\d+(?:\.\d+)?)", os.path.basename(fname))
        return float(m.group(1)) if m else None

    uv_list = []
    for f in files_uv:
        depth = extract_depth(f)
        try:
            df = pd.read_csv(f)
        except:
            continue
        df = df.rename(columns=lambda x: x.strip())
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df["Depth"] = depth
        uv_list.append(df[["Date","Depth","u","v"]])
    uv_all = pd.concat(uv_list, ignore_index=True) if uv_list else pd.DataFrame()

    ts_list = []
    for f in files_ts:
        depth = extract_depth(f)
        try:
            df = pd.read_csv(f)
        except:
            continue
        df = df.rename(columns=lambda x: x.strip())
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df["Depth"] = depth
        ts_list.append(df[["Date","Depth","t","s"]])
    ts_all = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame()

    if uv_all.empty and ts_all.empty:
        return pd.DataFrame()

    if uv_all.empty:
        merged = ts_all.copy()
        merged["u"] = np.nan
        merged["v"] = np.nan
    elif ts_all.empty:
        merged = uv_all.copy()
        merged["t"] = np.nan
        merged["s"] = np.nan
    else:
        merged = pd.merge(uv_all, ts_all, on=["Date","Depth"], how="outer")

    # JST
    merged["Date_JST"] = merged["Date"] + pd.Timedelta(hours=9)
    merged = merged.sort_values(["Date_JST","Depth"]).reset_index(drop=True)
    merged["Speed"] = np.sqrt(merged["u"].fillna(0)**2 + merged["v"].fillna(0)**2)
    merged["Direction_deg"] = (np.degrees(np.arctan2(merged["u"].fillna(0), merged["v"].fillna(0))) + 360) % 360
    return merged

# =========================================
# カレンダー用準備
# =========================================
def prepare_calendar(merged):
    if merged.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
    depths = np.sort(merged["Depth"].dropna().unique())
    merged = merged.copy()
    merged['Date_JST_rounded'] = merged['Date_JST'].dt.floor('h')
    merged['Date_str'] = merged['Date_JST_rounded'].dt.strftime('%m/%d')
    merged['Time_str'] = merged['Date_JST_rounded'].dt.strftime('%H:%M')
    calendar_t = merged.pivot_table(index='Depth', columns=['Date_str','Time_str'], values='t').reindex(depths)
    calendar_speed = merged.pivot_table(index=['Depth'], columns=['Date_str','Time_str'], values='Speed').reindex(depths)
    calendar_u = merged.pivot_table(index='Depth', columns=['Date_str','Time_str'], values='u').reindex(depths)
    calendar_v = merged.pivot_table(index='Depth', columns=['Date_str','Time_str'], values='v').reindex(depths)
    calendar_s = merged.pivot_table(index='Depth', columns=['Date_str','Time_str'], values='s').reindex(depths)
    cols = list(calendar_t.columns) if not calendar_t.empty else []
    return calendar_t, calendar_speed, calendar_u, calendar_v, calendar_s, cols, depths

# =========================================
# 3日平均（2024用）
# =========================================
def calc_periodic_avg_shortlabel(merged, period_days=3):
    if merged.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
    merged = merged.copy()
    merged['Date_day'] = merged['Date_JST'].dt.floor('d')
    unique_days = sorted(merged['Date_day'].unique())
    dfs = []
    labels = []
    for i in range(0, len(unique_days), period_days):
        block = unique_days[i:i+period_days]
        block_df = merged[merged['Date_day'].isin(block)]
        if block_df.empty:
            continue
        avg = block_df.groupby('Depth')[['t','s','u','v','Speed']].mean().reset_index()
        start = block[0]
        end = block[-1]
        if start.month == end.month:
            end_str = end.strftime('%d')
        else:
            end_str = end.strftime('%m/%d')
        label = f"{start.strftime('%m/%d')}-{end_str}" if len(block)>1 else start.strftime('%m/%d')
        avg['Period_label'] = label
        dfs.append(avg)
        labels.append(label)
    if not dfs:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [])
    concat = pd.concat(dfs, ignore_index=True)
    depths = np.sort(concat['Depth'].unique())
    T = concat.pivot(index='Depth', columns='Period_label', values='t').reindex(depths)
    Speed = concat.pivot(index='Depth', columns='Period_label', values='Speed').reindex(depths)
    U = concat.pivot(index='Depth', columns='Period_label', values='u').reindex(depths)
    V = concat.pivot(index='Depth', columns='Period_label', values='v').reindex(depths)
    S = concat.pivot(index='Depth', columns='Period_label', values='s').reindex(depths)
    return T, Speed, U, V, S, labels, depths

# =========================================
# 表描画
# =========================================
def plot_table(ax, T, Speed, U, V, S, cols, depths, t_min=10, t_max=25):
    if T is None or T.empty:
        ax.text(0.5,0.5,"No data", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    n_rows, n_cols = T.shape
    Speed_kt = Speed * 1.94384
    aspect_ratio = n_cols / n_rows if n_rows>0 else 1
    fig_width = 16
    fig_height = max(6, fig_width / max(aspect_ratio, 0.1))
    ax.figure.set_size_inches(fig_width, fig_height)
    norm_t = mcolors.Normalize(vmin=t_min, vmax=t_max)
    ax.pcolormesh(np.arange(n_cols+1), np.arange(n_rows+1), T.iloc[::-1, :],
                  cmap=plt.cm.coolwarm, norm=norm_t, shading='auto')
    for i in range(n_rows+1):
        ax.axhline(i, color='k', lw=0.4)
    for j in range(n_cols+1):
        ax.axvline(j, color='k', lw=0.4)
    s_min, s_max = 32, 35
    norm_s = mcolors.BoundaryNorm(np.arange(s_min, s_max+0.1, 0.1), plt.cm.Spectral_r.N)
    cmap_speed = plt.colormaps.get("viridis")
    norm_speed = mcolors.Normalize(vmin=0, vmax=np.nanmax(Speed_kt.values) if np.any(~np.isnan(Speed_kt.values)) else 1.0)
    arrow_scale = 0.5
    for i in range(n_rows):
        row_idx = n_rows - i - 1
        for j in range(n_cols):
            try:
                u_val = U.iloc[i,j]
                v_val = V.iloc[i,j]
                sp_val_kt = Speed_kt.iloc[i,j]
                t_val = T.iloc[i,j]
                s_val = S.iloc[i,j]
            except Exception:
                u_val = v_val = sp_val_kt = t_val = s_val = np.nan
            if not np.isnan(s_val):
                ax.pcolormesh([j, j+1], [row_idx, row_idx+0.20],
                              np.array([[s_val]]), cmap=plt.cm.Spectral_r, norm=norm_s, shading='auto')
                ax.text(j+0.5, row_idx+0.10, f"{s_val:.1f}‰", ha='center', va='center', fontsize=11, color='black')
            if not np.isnan(sp_val_kt):
                text_y_speed = row_idx + 0.20 + 0.25
                ax.text(j+0.5, text_y_speed, f"{sp_val_kt:.1f} kt", ha='center', va='bottom', fontsize=12, color='black')
            if not np.isnan(t_val):
                text_y_temp = row_idx + 0.20 + 0.03
                ax.text(j+0.5, text_y_temp, f"{t_val:.1f}°C", ha='center', va='bottom', fontsize=12, color='black')
            if not np.isnan(u_val) and not np.isnan(v_val):
                arrow_y = row_idx + 0.20 + 0.25 + 0.35
                color = cmap_speed(norm_speed(sp_val_kt if not np.isnan(sp_val_kt) else 0.0))
                arrow_scale = 0.2
                u_plot = u_val * arrow_scale
                v_plot = v_val * arrow_scale
                mag = np.sqrt(u_plot**2 + v_plot**2)
                min_arrow_len = 0.15
                max_arrow_len = 0.4
                if mag > max_arrow_len:
                    factor = max_arrow_len / mag
                    u_plot *= factor
                    v_plot *= factor
                elif mag < min_arrow_len:
                    factor = min_arrow_len / (mag if mag>0 else 1)
                    u_plot *= factor
                    v_plot *= factor
                ax.quiver(j+0.5, arrow_y, u_plot, v_plot,
                          angles='xy', scale_units='xy', scale=1,
                          width=0.006, headwidth=5, headlength=7, headaxislength=6,
                          color=color, pivot='middle')
    max_labels = 15
    step = max(1, n_rows // max_labels)
    yticks = np.arange(0, n_rows, step) + 0.5
    ylabels = np.round(depths[::-1][::step], 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_ylabel("水深 [m]", fontsize=14)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols)+0.5)
    ax.set_xticklabels([])
    if not cols:
        return
    if isinstance(cols[0], tuple):
        cols_tuples = cols
    else:
        cols_tuples = [(str(c), "") for c in cols]
    unique_dates = []
    date_positions = []
    for j, (date_str, time_str) in enumerate(cols_tuples):
        if date_str not in unique_dates:
            unique_dates.append(date_str)
            date_positions.append(j)
        if time_str:
            ax.text(j+0.5, n_rows+0.05, time_str, ha='center', va='bottom', fontsize=12)
    for date_str, pos in zip(unique_dates, date_positions):
        ax.text(pos+0.5, n_rows+0.25, date_str, ha='center', va='bottom', fontsize=12, fontweight='bold')

# =========================================
# Streamlit UI
# パスワード保護（最小限）
# =========================================
PASSWORD = "1234"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.pwd_input = ""

if not st.session_state.authenticated:
    st.session_state.pwd_input = st.text_input("パスワードを入力", type="password", value=st.session_state.pwd_input)

    if st.button("ログイン"):
        if st.session_state.pwd_input == PASSWORD:
            st.session_state.authenticated = True
        else:
            st.error("パスワードが違います")

    if not st.session_state.authenticated:
        st.stop()  # 認証されるまで以降処理を停止

# ここから下は認証済みのみ実行される
parent_folder = "data"
subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
selected_subfolder = st.selectbox("", options=subfolders)

# 2025_now
folder_now = os.path.join(parent_folder, selected_subfolder, "2025_now")
if not os.path.exists(folder_now):
    st.error(f"{folder_now} が見つかりません")
else:
    merged_now = load_uv_ts(folder_now)
    if merged_now.empty:
        st.warning(f"{folder_now} に読み込める CSV がありません。")
    else:
        T_now, Speed_now, U_now, V_now, S_now, cols_now, depths_now = prepare_calendar(merged_now)
        selected_dates = sorted(list({c[0] for c in cols_now})) if cols_now else []
        mode = st.radio("短期予測", ["12時間毎（7日間）", "1時間毎"])
        cols_filtered = []

        if mode == "12時間毎（7日間）":
            for col in cols_now:
                date_str, time_str = col
                hour = int(time_str.split(":")[0])
                col_date = pd.to_datetime(date_str, format="%m/%d")
                if hour in [0,12]:
                    if selected_dates:
                        min_date = pd.to_datetime(selected_dates[0], format="%m/%d")
                        max_date = min_date + pd.Timedelta(days=8)
                        if min_date <= col_date <= max_date:
                            cols_filtered.append(col)
                    else:
                        cols_filtered.append(col)
        else:  # 1時間毎
            # カレンダーで選択
            min_date = merged_now['Date_JST'].min().date()
            max_date = merged_now['Date_JST'].max().date()
            selected_date = st.date_input("日付を選択", value=min_date, min_value=min_date, max_value=max_date)
            # 3時〜17時まで1時間毎
            hours = range(3,18)
            date_str = selected_date.strftime("%m/%d")
            for h in hours:
                time_str = f"{h:02d}:00"
                if (date_str, time_str) in cols_now:
                    cols_filtered.append((date_str, time_str))

        if not cols_filtered:
            st.warning("選択された日付・時間に該当するデータがありません。")
        else:
            T_filtered = T_now.loc[:, cols_filtered]
            Speed_filtered = Speed_now.loc[:, cols_filtered]
            U_filtered = U_now.loc[:, cols_filtered]
            V_filtered = V_now.loc[:, cols_filtered]
            S_filtered = S_now.loc[:, cols_filtered]
            fig, ax = plt.subplots(dpi=120)
            plot_table(ax, T_filtered, Speed_filtered, U_filtered, V_filtered, S_filtered, cols_filtered, depths_now, t_min=10, t_max=25)
            st.pyplot(fig, use_container_width=True)

# 2024
show_2024 = st.checkbox("参考：2024 (3日平均)")
if show_2024:
    period_days = 3
    folder_2024 = os.path.join(parent_folder, selected_subfolder, "2024")
    if not os.path.exists(folder_2024):
        st.warning(f"{folder_2024} が見つかりません")
    else:
        merged_2024 = load_uv_ts(folder_2024)
        if merged_2024.empty:
            st.warning(f"{folder_2024} に読み込める CSV がありません。")
        else:
            T_2024, Speed_2024, U_2024, V_2024, S_2024, cols_2024, depths_2024 = calc_periodic_avg_shortlabel(merged_2024, period_days=period_days)
            if T_2024.empty:
                st.warning("2024 の期間平均でデータが作れませんでした。")
            else:
                fig2, ax2 = plt.subplots(dpi=120)
                plot_table(ax2, T_2024, Speed_2024, U_2024, V_2024, S_2024, cols_2024, depths_2024, t_min=10, t_max=25)
                st.pyplot(fig2, use_container_width=True)