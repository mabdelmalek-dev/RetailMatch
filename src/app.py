# src/app.py
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

# If your templates/static are inside src/, this default works:
app = Flask(__name__)

# ---- Load artifacts produced by train_content.py ----
df = pickle.load(open("data/products.pkl", "rb"))
vectorizer = pickle.load(open("data/vectorizer.pkl", "rb"))
tfidf_matrix = pickle.load(open("data/tfidf_matrix.pkl", "rb"))

# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.get("/similar")
def similar():
    product_id = request.args.get("product_id", "").strip()
    k = int(request.args.get("k", 5))
    if not product_id or product_id not in df["product_id"].values:
        return jsonify({"error": "Product ID not found or missing"}), 404

    idx = df.index[df["product_id"] == product_id][0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[-(k+1):][::-1]  # top k + self
    items = df.iloc[order[1:k+1]][["product_id", "title", "brand", "price", "category"]]
    return jsonify({"query": product_id, "results": items.to_dict(orient="records")})

@app.get("/top")
def top():
    category = request.args.get("category", "").strip()
    k = int(request.args.get("k", 5))
    if not category:
        return jsonify({"error": "category is required"}), 400

    # Partial, case-insensitive match (so "TV" matches "Television", etc.)
    sub = df[df["category"].str.lower().str.contains(category.lower(), na=False)]
    if sub.empty:
        return jsonify({"error": f"No products found for category '{category}'"}), 404

    sub = sub.sort_values("price", ascending=False).head(k)
    items = sub[["product_id", "title", "brand", "price", "category"]]
    return jsonify({"category": category, "results": items.to_dict(orient="records")})

# Quick debug route to see what's registered
@app.get("/__routes")
def routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

if __name__ == "__main__":
    app.run(debug=True)
