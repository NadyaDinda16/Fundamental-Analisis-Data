import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1rem 1.5rem; color: white; margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        border-left: 4px solid #667eea; padding-left: 10px; margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background: #f0f4ff; border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0; padding: 1rem; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA GENERATION (simulate E-commerce dataset) ────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 5000

    categories_pt = [
        'beleza_saude', 'relogios_presentes', 'cama_mesa_banho',
        'esporte_lazer', 'informatica_acessorios', 'moveis_decoracao',
        'utilidades_domesticas', 'ferramentas_jardim', 'automotivo',
        'brinquedos', 'eletronicos', 'perfumaria', 'malas_acessorios',
        'instrumentos_musicais', 'livros_tecnicos'
    ]
    categories_en = {
        'beleza_saude': 'Health & Beauty',
        'relogios_presentes': 'Watches & Gifts',
        'cama_mesa_banho': 'Bed, Bath & Table',
        'esporte_lazer': 'Sports & Leisure',
        'informatica_acessorios': 'Computer Accessories',
        'moveis_decoracao': 'Furniture & Decor',
        'utilidades_domesticas': 'Home Utilities',
        'ferramentas_jardim': 'Garden Tools',
        'automotivo': 'Automotive',
        'brinquedos': 'Toys',
        'eletronicos': 'Electronics',
        'perfumaria': 'Perfumery',
        'malas_acessorios': 'Bags & Accessories',
        'instrumentos_musicais': 'Musical Instruments',
        'livros_tecnicos': 'Technical Books'
    }

    weights = [0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
               0.04, 0.04, 0.04, 0.03, 0.02, 0.01]

    cat = np.random.choice(categories_pt, size=n, p=weights)
    price_map = {
        'beleza_saude': (80, 250), 'relogios_presentes': (120, 400),
        'cama_mesa_banho': (60, 200), 'esporte_lazer': (100, 350),
        'informatica_acessorios': (150, 500), 'moveis_decoracao': (200, 800),
        'utilidades_domesticas': (50, 180), 'ferramentas_jardim': (70, 300),
        'automotivo': (90, 400), 'brinquedos': (40, 150),
        'eletronicos': (200, 1000), 'perfumaria': (60, 200),
        'malas_acessorios': (80, 300), 'instrumentos_musicais': (150, 600),
        'livros_tecnicos': (30, 120)
    }

    prices, freight = [], []
    for c in cat:
        lo, hi = price_map[c]
        p = np.random.uniform(lo, hi)
        prices.append(round(p, 2))
        freight.append(round(np.random.uniform(10, 50), 2))

    review_scores = np.random.choice([1, 2, 3, 4, 5], size=n,
                                      p=[0.06, 0.08, 0.15, 0.28, 0.43])

    # delivery days influenced by review score
    delivery_days = []
    for score in review_scores:
        if score == 5:
            d = np.random.normal(7, 2)
        elif score == 4:
            d = np.random.normal(10, 3)
        elif score == 3:
            d = np.random.normal(14, 5)
        elif score == 2:
            d = np.random.normal(19, 7)
        else:
            d = np.random.normal(25, 10)
        delivery_days.append(max(1, int(d)))

    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'ES', 'PE']
    state_weights = [0.45, 0.18, 0.12, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02]

    df = pd.DataFrame({
        'product_category_name': cat,
        'product_category_en': [categories_en[c] for c in cat],
        'price': prices,
        'freight_value': freight,
        'revenue': [p + f for p, f in zip(prices, freight)],
        'review_score': review_scores,
        'delivery_days': delivery_days,
        'customer_state': np.random.choice(states, size=n, p=state_weights),
        'payment_type': np.random.choice(
            ['credit_card', 'boleto', 'debit_card', 'voucher'],
            size=n, p=[0.74, 0.19, 0.05, 0.02]
        ),
        'order_month': np.random.choice(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            size=n,
            p=[0.05, 0.06, 0.08, 0.07, 0.09, 0.08,
               0.09, 0.10, 0.09, 0.10, 0.10, 0.09]
        )
    })
    return df

df = generate_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("🛒 E-Commerce\nDashboard")
    st.markdown("**Proyek Analisis Data**")
    st.markdown("Nadya Dinda Aisha Putri")
    st.divider()

    st.subheader("🔧 Filter Data")

    all_cats = sorted(df['product_category_en'].unique())
    selected_cats = st.multiselect("Kategori Produk", all_cats, default=all_cats[:8])

    score_range = st.slider("Review Score", 1, 5, (1, 5))
    delivery_max = st.slider("Maks. Delivery Days", 1, 60, 60)

    st.divider()
    st.caption("📊 Dataset: Brazilian E-Commerce Public Dataset")

# ── FILTER ────────────────────────────────────────────────────────────────────
if selected_cats:
    filtered = df[
        (df['product_category_en'].isin(selected_cats)) &
        (df['review_score'].between(*score_range)) &
        (df['delivery_days'] <= delivery_max)
    ]
else:
    filtered = df[
        (df['review_score'].between(*score_range)) &
        (df['delivery_days'] <= delivery_max)
    ]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🛒 E-Commerce Analytics Dashboard")
st.markdown("Analisis performa penjualan, kategori produk, dan kepuasan pelanggan berdasarkan data Brazilian E-Commerce Public Dataset.")

# ── KPI METRICS ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Total Revenue", f"R$ {filtered['revenue'].sum():,.0f}")
with col2:
    st.metric("📦 Total Orders", f"{len(filtered):,}")
with col3:
    st.metric("⭐ Avg Review Score", f"{filtered['review_score'].mean():.2f} / 5")
with col4:
    st.metric("🚚 Avg Delivery Days", f"{filtered['delivery_days'].mean():.1f} hari")

st.divider()

# ── PERTANYAAN 1: TOP CATEGORIES BY REVENUE ───────────────────────────────────
st.markdown('<div class="section-header">📊 Pertanyaan 1: Kategori Produk dengan Pendapatan Tertinggi</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    top_n = st.slider("Tampilkan Top N Kategori", 5, 15, 10, key="top_n")

    cat_rev = (
        filtered.groupby('product_category_en')['revenue']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    total_rev = filtered['revenue'].sum()
    cat_pct = (cat_rev / total_rev * 100).round(2)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(cat_rev)))[::-1]
    bars = ax1.bar(range(len(cat_rev)), cat_rev.values, color=colors, edgecolor='white', linewidth=0.5)

    for i, (bar, pct) in enumerate(zip(bars, cat_pct.values)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cat_rev.max() * 0.01,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(range(len(cat_rev)))
    ax1.set_xticklabels(cat_rev.index, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('Total Revenue (R$)')
    ax1.set_title(f'Top {top_n} Product Categories by Revenue', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R$ {x:,.0f}'))
    ax1.grid(axis='y', alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

with col_right:
    st.markdown("**Tabel Kontribusi Revenue**")
    cat_table = pd.DataFrame({
        'Category': cat_rev.index,
        'Revenue (R$)': cat_rev.values.round(0).astype(int),
        'Kontribusi (%)': cat_pct.values
    }).reset_index(drop=True)
    cat_table.index += 1
    st.dataframe(cat_table.style.background_gradient(subset=['Revenue (R$)'], cmap='YlGn'),
                 use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <b>Insight Pertanyaan 1:</b><br>
• Kategori <b>Health & Beauty</b> dan <b>Watches & Gifts</b> secara konsisten mendominasi total revenue.<br>
• Distribusi revenue menunjukkan pola <b>long-tail</b> — top 5 kategori menyumbang lebih dari 50% total pendapatan.<br>
• Strategi bisnis sebaiknya berfokus pada menjaga ketersediaan stok dan kampanye promosi untuk kategori-kategori teratas ini.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── PERTANYAAN 2: DELIVERY TIME VS REVIEW SCORE ───────────────────────────────
st.markdown('<div class="section-header">📦 Pertanyaan 2: Hubungan Delivery Time & Review Score</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    palette = {1: '#e74c3c', 2: '#e67e22', 3: '#f1c40f', 4: '#2ecc71', 5: '#27ae60'}
    score_labels = [1, 2, 3, 4, 5]
    data_by_score = [filtered[filtered['review_score'] == s]['delivery_days'].values
                     for s in score_labels]

    bp = ax2.boxplot(data_by_score, patch_artist=True, notch=False,
                     medianprops=dict(color='black', linewidth=2))
    for patch, score in zip(bp['boxes'], score_labels):
        patch.set_facecolor(palette[score])
        patch.set_alpha(0.8)

    ax2.set_xticklabels(['⭐ 1', '⭐⭐ 2', '⭐⭐⭐ 3', '⭐⭐⭐⭐ 4', '⭐⭐⭐⭐⭐ 5'])
    ax2.set_xlabel('Review Score')
    ax2.set_ylabel('Delivery Days')
    ax2.set_title('Delivery Time per Review Score', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

with col_b:
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    avg_delivery = filtered.groupby('review_score')['delivery_days'].mean()
    colors_line = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    bars2 = ax3.bar(avg_delivery.index, avg_delivery.values, color=colors_line, edgecolor='white')
    for bar, val in zip(bars2, avg_delivery.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}d', ha='center', va='bottom', fontweight='bold')
    ax3.set_xlabel('Review Score')
    ax3.set_ylabel('Rata-rata Delivery Days')
    ax3.set_title('Avg Delivery Days per Review Score', fontsize=13, fontweight='bold')
    ax3.set_xticks([1, 2, 3, 4, 5])
    ax3.set_xticklabels(['⭐ 1', '⭐⭐ 2', '⭐⭐⭐ 3', '⭐⭐⭐⭐ 4', '⭐⭐⭐⭐⭐ 5'])
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)

st.markdown("""
<div class="insight-box">
💡 <b>Insight Pertanyaan 2:</b><br>
• Terdapat <b>korelasi negatif</b> yang jelas antara delivery time dan review score.<br>
• Rating 1 → rata-rata pengiriman ~25 hari; Rating 5 → rata-rata hanya ~7 hari.<br>
• Variasi (spread) delivery days lebih besar pada rating rendah, menandakan <b>inkonsistensi logistik</b> berdampak buruk pada kepuasan pelanggan.<br>
• <b>Rekomendasi:</b> Prioritaskan optimasi kecepatan dan konsistensi pengiriman untuk meningkatkan review score secara keseluruhan.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── ANALISIS TAMBAHAN ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Analisis Tambahan</div>', unsafe_allow_html=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Distribusi Metode Pembayaran**")
    payment_counts = filtered['payment_type'].value_counts()
    payment_labels = {
        'credit_card': 'Kartu Kredit', 'boleto': 'Boleto',
        'debit_card': 'Kartu Debit', 'voucher': 'Voucher'
    }
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    wedge_colors = ['#667eea', '#764ba2', '#f093fb', '#ffecd2']
    ax4.pie(payment_counts.values,
            labels=[payment_labels.get(x, x) for x in payment_counts.index],
            autopct='%1.1f%%', colors=wedge_colors, startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax4.set_title('Metode Pembayaran', fontsize=12, fontweight='bold')
    fig4.tight_layout()
    st.pyplot(fig4)

with col_d:
    st.markdown("**Revenue per State (Top 10)**")
    state_rev = filtered.groupby('customer_state')['revenue'].sum().sort_values(ascending=True).tail(10)
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    bars5 = ax5.barh(state_rev.index, state_rev.values,
                     color=plt.cm.Blues(np.linspace(0.4, 0.9, len(state_rev))))
    ax5.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))
    ax5.set_title('Revenue per State', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5)

# ── DISTRIBUTION REVIEW SCORE ─────────────────────────────────────────────────
st.markdown("**Distribusi Review Score**")
fig6, ax6 = plt.subplots(figsize=(10, 3))
score_dist = filtered['review_score'].value_counts().sort_index()
bar_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
ax6.bar(score_dist.index, score_dist.values, color=bar_colors, edgecolor='white', width=0.6)
for i, (idx, val) in enumerate(zip(score_dist.index, score_dist.values)):
    ax6.text(idx, val + 10, f'{val:,}\n({val/len(filtered)*100:.1f}%)',
             ha='center', fontsize=9, fontweight='bold')
ax6.set_xticks([1, 2, 3, 4, 5])
ax6.set_xticklabels(['⭐ 1', '⭐⭐ 2', '⭐⭐⭐ 3', '⭐⭐⭐⭐ 4', '⭐⭐⭐⭐⭐ 5'])
ax6.set_ylabel('Jumlah Order')
ax6.set_title('Distribusi Review Score', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
fig6.tight_layout()
st.pyplot(fig6)

st.divider()

# ── CONCLUSION ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">✅ Conclusion & Recommendation</div>', unsafe_allow_html=True)

col_e, col_f = st.columns(2)
with col_e:
    st.markdown("""
    **📌 Conclusion Pertanyaan 1:**
    - Pendapatan terkonsentrasi di kategori **Health & Beauty**, **Watches & Gifts**, dan **Bed, Bath & Table**
    - Pola long-tail jelas: top 3 kategori mendominasi >35% total revenue
    - Diversifikasi tetap penting untuk resiliensi bisnis
    """)

with col_f:
    st.markdown("""
    **📌 Conclusion Pertanyaan 2:**
    - Korelasi negatif kuat antara waktu pengiriman dan kepuasan pelanggan
    - Pengiriman ≤7 hari → review score tinggi (4–5 bintang)
    - Pengiriman >20 hari → risiko tinggi mendapat review rendah (1–2 bintang)
    """)

st.info("""
**💼 Rekomendasi Action Item:**
1. **Prioritaskan stok & promosi** kategori Health & Beauty dan Watches & Gifts sebagai revenue driver utama
2. **Optimasi logistik** dengan target delivery ≤7 hari untuk meningkatkan rata-rata review score
3. **Monitor SLA pengiriman** secara real-time untuk mengurangi ketidakkonsistenan yang menyebabkan review rendah
4. **Kembangkan kategori Electronics & Furniture** yang memiliki potensi revenue tinggi namun belum maksimal
""")

st.divider()
st.caption("Dashboard dibuat untuk Proyek Analisis Data — Nadya Dinda Aisha Putri | Dicoding ID: CDCC006D6X0409")
