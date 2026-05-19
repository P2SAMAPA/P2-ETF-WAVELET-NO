# Wavelet Neural Operator (WNO) Engine

Implements a neural operator using a multi‑scale 1D CNN that mimics a wavelet decomposition. The operator learns a mapping from a sequence of past log returns to the next day's return. Wavelets capture both frequency and time localisation, making the model suitable for non‑stationary financial returns.

- **Operator architecture:** Multi‑scale 1D CNN (wavelet‑like)
- **Input:** Sequence of daily log returns (length `SEQ_LEN`)
- **Training:** Sliding windows on a rolling window (63‑2016 days)
- **Output:** predicted next‑day return for each ETF
- **Multi‑window evaluation:** picks the window giving the highest predicted return per ETF
- **Ranking:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
