import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #4361ee;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #eef2ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #1a1a2e;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .rec-box {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #166534;
        font-size: 0.92rem;
        line-height: 1.7;
        border-left: 4px solid #22c55e;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FIND DATASET FOLDER
# ─────────────────────────────────────────────
def find_data_path():
    folder_name = "E-commerce-public-dataset"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    candidates = [
        os.path.join(script_dir, folder_name),
        os.path.join(os.getcwd(), folder_name),
        os.path.join(script_dir, "..", folder_name),
        f"/mount/src/fundamental-analisis-data/{folder_name}",
        f"/mount/src/proyek-analisis-data/{folder_name}",
        os.path.join(os.path.expanduser("~"), folder_name),
    ]

    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isdir(path) and any(f.endswith(".csv") for f in os.listdir(path)):
            return path
    return None


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(data_path: str):
    data = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            key = file.replace(".csv", "")
            data[key] = pd.read_csv(os.path.join(data_path, file))

    required = [
        'customers_dataset', 'orders_dataset', 'order_items_dataset',
        'products_dataset', 'order_reviews_dataset', 'order_payments_dataset'
    ]
    missing = [r for r in required if r not in data]
    if missing:
        raise FileNotFoundError(
            f"File CSV berikut tidak ditemukan di folder dataset: {missing}"
        )

    customers     = data['customers_dataset']
    orders        = data['orders_dataset']
    order_items   = data['order_items_dataset']
    products      = data['products_dataset']
    order_reviews = data['order_reviews_dataset']
    payments      = data['order_payments_dataset']

    products['product_category_name'] = products['product_category_name'].fillna('unknown')
    orders = orders.dropna(subset=['order_purchase_timestamp'])
    order_reviews = order_reviews.drop_duplicates(subset=['review_id'])
    order_items   = order_items.drop_duplicates()
    orders        = orders.drop_duplicates()

    orders['order_purchase_timestamp']      = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])

    orders_delivered = orders[orders['order_status'] == 'delivered'].copy()
    orders_delivered = orders_delivered.dropna(subset=['order_delivered_customer_date'])
    orders_delivered['delivery_days'] = (
        orders_delivered['order_delivered_customer_date'] -
        orders_delivered['order_purchase_timestamp']
    ).dt.days

    order_items = order_items[
        (order_items['price'] >= 0) & (order_items['freight_value'] >= 0)
    ]

    main_df = (
        orders_delivered
        .merge(order_items,   on='order_id')
        .merge(products,      on='product_id')
        .merge(payments,      on='order_id')
        .merge(customers,     on='customer_id')
    )
    main_df['revenue'] = main_df['price'] + main_df['freight_value']

    main_df = main_df.merge(
        order_reviews[['order_id', 'review_score']],
        on='order_id', how='left'
    )

    category_map = {
        'beleza_saude': 'Health & Beauty',
        'relogios_presentes': 'Watches & Gifts',
        'cama_mesa_banho': 'Bed Bath & Table',
        'esporte_lazer': 'Sports & Leisure',
        'informatica_acessorios': 'Computer Accessories',
        'moveis_decoracao': 'Furniture & Decor',
        'utilidades_domesticas': 'Housewares',
        'ferramentas_jardim': 'Garden Tools',
        'telefonia': 'Telephony',
        'automotivo': 'Auto',
        'brinquedos': 'Toys',
        'cool_stuff': 'Cool Stuff',
        'eletronicos': 'Electronics',
        'eletrodomesticos': 'Appliances',
        'unknown': 'Unknown',
    }
    main_df['category_en'] = main_df['product_category_name'].map(
        lambda x: category_map.get(x, x.replace('_', ' ').title())
    )

    return main_df


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🛒 E-Commerce Public Dataset Dashboard")
st.markdown(
    "Analisis **Revenue Kategori Produk** & "
    "**Pengaruh Waktu Pengiriman Terhadap Kepuasan Pelanggan** (2017–2018)"
)

# ─────────────────────────────────────────────
# DETECT & LOAD DATA
# ─────────────────────────────────────────────
data_path = find_data_path()

if data_path is None:
    st.error("Folder dataset `E-commerce-public-dataset` tidak ditemukan!")
try:
    with st.spinner("Memuat data..."):
        main_df = load_data(data_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────────────────────
st.sidebar.header("Filter")

top_n = st.sidebar.slider("Jumlah Kategori (Q1)", 5, 20, 10)

score_filter = st.sidebar.multiselect(
    "Filter Review Score (Q2)",
    options=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5]
)

year_options = sorted(main_df['order_purchase_timestamp'].dt.year.unique().tolist())
if len(year_options) == 1:
    year_options = year_options * 2
year_range = st.sidebar.select_slider(
    "Filter Tahun",
    options=year_options,
    value=(year_options[0], year_options[-1])
)

df_filtered = main_df[
    main_df['order_purchase_timestamp'].dt.year.between(year_range[0], year_range[1])
]

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
total_rev    = df_filtered['revenue'].sum()
total_orders = df_filtered['order_id'].nunique()
avg_delivery = df_filtered['delivery_days'].mean()
avg_score    = df_filtered['review_score'].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue",       f"R$ {total_rev:,.0f}")
c2.metric("📦 Total Orders",        f"{total_orders:,}")
c3.metric("🚚 Avg Delivery (hari)", f"{avg_delivery:.1f}")
c4.metric("⭐ Avg Review Score",    f"{avg_score:.2f}")

st.markdown("---")


# ═══════════════════════════════════════════════════
# Q1 – REVENUE PER KATEGORI
# ═══════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Pertanyaan 1 — Kategori Produk dengan Revenue Tertinggi</div>',
            unsafe_allow_html=True)

category_revenue = (
    df_filtered.groupby('category_en')['revenue']
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .reset_index()
)
category_revenue.columns = ['Kategori', 'Revenue']
category_revenue['Kontribusi (%)'] = (
    category_revenue['Revenue'] / df_filtered['revenue'].sum() * 100
).round(2)

col_left, col_right = st.columns([3, 2])

with col_left:
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("Blues_r", top_n)
    bars = ax1.barh(
        category_revenue['Kategori'][::-1],
        category_revenue['Revenue'][::-1],
        color=colors[::-1],
        edgecolor='white'
    )
    grand_total = df_filtered['revenue'].sum()
    for bar, val in zip(bars, category_revenue['Revenue'][::-1]):
        ax1.text(
            bar.get_width() + grand_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"R$ {val/1e6:.2f}M",
            va='center', fontsize=8.5, color='#333'
        )
    ax1.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'R$ {x/1e6:.1f}M')
    )
    ax1.set_xlabel("Total Revenue")
    ax1.set_title(f"Top {top_n} Kategori Produk berdasarkan Revenue",
                  fontsize=13, fontweight='bold', pad=12)
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)

with col_right:
    st.markdown("**Tabel Ringkasan**")
    st.dataframe(
        category_revenue.style.format({
            'Revenue': 'R$ {:,.0f}',
            'Kontribusi (%)': '{:.2f}%'
        }),
        use_container_width=True,
        height=380
    )

# Pie chart
fig2, ax2 = plt.subplots(figsize=(6, 4))
top5 = category_revenue.head(5)
others_rev = df_filtered['revenue'].sum() - top5['Revenue'].sum()
pie_labels = list(top5['Kategori']) + ['Others']
pie_values = list(top5['Revenue']) + [others_rev]
palette_pie = sns.color_palette("Set2", len(pie_labels))
wedges, texts, autotexts = ax2.pie(
    pie_values, labels=pie_labels, autopct='%1.1f%%',
    colors=palette_pie, startangle=140, pctdistance=0.82,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title("Kontribusi Revenue — Top 5 Kategori vs Lainnya",
              fontsize=11, fontweight='bold')
plt.tight_layout()

col_pie, col_ins = st.columns([2, 3])
with col_pie:
    st.pyplot(fig2)
with col_ins:
    st.markdown('<div class="insight-box">💡 <b>Insight:</b><br>'
                '• Kategori <b>Health & Beauty</b> menjadi kontributor revenue terbesar.<br>'
                '• 5 kategori teratas menyumbang hampir <b>40% total revenue</b>.<br>'
                '• Produk kebutuhan sehari-hari & lifestyle mendominasi pasar.<br>'
                '• Watches & Gifts dan Bed Bath & Table potensial untuk <i>upselling</i>.</div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="rec-box">✅ <b>Rekomendasi:</b><br>'
                '• <b>Scaling:</b> Prioritaskan Health & Beauty dengan promo & iklan.<br>'
                '• <b>Upselling:</b> Bundle deals untuk Watches & Gifts dan Bed Bath & Table.<br>'
                '• <b>Exposure:</b> Tingkatkan visibilitas kategori menengah (Sports, Computer Accessories).</div>',
                unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════
# Q2 – DELIVERY TIME VS REVIEW SCORE
# ═══════════════════════════════════════════════════
st.markdown('<div class="section-header">🚚 Pertanyaan 2 — Pengaruh Delivery Time terhadap Review Score</div>',
            unsafe_allow_html=True)

df_q2 = df_filtered[df_filtered['review_score'].isin(score_filter)].dropna(
    subset=['review_score', 'delivery_days']
).copy()
df_q2['review_score'] = df_q2['review_score'].astype(int)

palette_box = {
    '1': '#ef4444',
    '2': '#f97316',
    '3': '#eab308',
    '4': '#84cc16',
    '5': '#22c55e'
}

col_box, col_bar = st.columns(2)

with col_box:
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=df_q2, x='review_score', y='delivery_days',
        palette=palette_box, width=0.55,
        flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3},
        ax=ax3
    )
    ax3.set_title("Distribusi Delivery Time per Review Score",
                  fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Review Score")
    ax3.set_ylabel("Delivery Days")
    ax3.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

with col_bar:
    avg_del_score = (
        df_q2.groupby('review_score')['delivery_days']
        .mean().reset_index()
    )
    avg_del_score.columns = ['Review Score', 'Avg Delivery Days']
    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    bar_colors = [palette_box.get(int(s), '#888') for s in avg_del_score['Review Score']]
    bars4 = ax4.bar(
        avg_del_score['Review Score'].astype(str),
        avg_del_score['Avg Delivery Days'],
        color=bar_colors, edgecolor='white', width=0.55
    )
    for bar, val in zip(bars4, avg_del_score['Avg Delivery Days']):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}d",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    ax4.set_title("Rata-rata Delivery Time per Review Score",
                  fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Review Score")
    ax4.set_ylabel("Avg Delivery Days")
    ax4.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig4)

# Scatter + trend
fig5, ax5 = plt.subplots(figsize=(10, 3.5))
sample = df_q2.sample(min(3000, len(df_q2)), random_state=42)
sns.regplot(
    data=sample, x='delivery_days', y='review_score',
    scatter_kws={'alpha': 0.15, 'color': '#4361ee', 's': 20},
    line_kws={'color': '#ef4444', 'linewidth': 2},
    ax=ax5
)
ax5.set_title("Trend: Delivery Time vs Review Score",
              fontsize=12, fontweight='bold', pad=10)
ax5.set_xlabel("Delivery Days")
ax5.set_ylabel("Review Score")
ax5.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig5)

col_ins2, col_rec2 = st.columns(2)
with col_ins2:
    st.markdown('<div class="insight-box">💡 <b>Insight:</b><br>'
                '• Terdapat <b>hubungan negatif</b> antara waktu pengiriman dan review score.<br>'
                '• Selisih rata-rata delivery time antara bintang 1 vs bintang 5 mencapai <b>sekitar 9 hari</b>.<br>'
                '• Review bintang 5 memiliki median delivery time <b>paling rendah</b>.<br>'
                '• Keterlambatan pengiriman berkontribusi langsung terhadap penurunan kepuasan.</div>',
                unsafe_allow_html=True)
with col_rec2:
    st.markdown('<div class="rec-box">✅ <b>Rekomendasi:</b><br>'
                '• 🚀 <b>Optimasi logistik:</b> Target delivery kurang dari 10–12 hari.<br>'
                '• 📍 <b>Prioritas area demand tinggi:</b> Percepat pengiriman ke kota utama.<br>'
                '• 📊 <b>Monitoring SLA:</b> Identifikasi order berpotensi terlambat lebih awal.<br>'
                '• 🤝 <b>Seleksi ekspedisi:</b> Pilih partner dengan performa terbaik.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "E-Commerce Public Dataset Analysis 2017–2018"
    "</div>",
    unsafe_allow_html=True
)
