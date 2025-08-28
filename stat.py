import joblib

# Load later
clf = joblib.load("tagging_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# ----------------------------
# 7. Inference: predict tags for a new video
# ----------------------------
new_videos = ["Liga Portugal : William Gomes et le FC Porto continuent leur parcours parfait",
             "L'engouement pour la WNBA",
              "Champions Cup: le Stade Toulousain décimé, l'occasion rêvée pour l'UBB",
              "Champions Cup: le Stade Toulousain pour l'UBB",
              "A Philly, Guerschon Yabusele",
              "Lilo & Stitch, L’histoire touchante et drôle d’une petite fille hawaïenne solitaire et d’un extra-terrestre fugitif qui l’aide à renouer le lien avec sa famille.",
              "Freaky Friday dans la peau de ma mère, Veuve sur le point de se remarier, Tess et sa fille Anna ne s'entendent pas. Un jeudi, leur rancoeur éclate. Deux biscuits vont tout compliquer en créant un choc mystique. Le lendemain, Tess et Anna se retrouvent dans le corps l'une de l'autre...",
              "Astérix et Obélix : Mission Cléopâtre, Cléopâtre décide, pour défier l'Empereur romain Jules César, de construire en trois mois un palais somptueux en plein désert. Si elle y parvient, celui-ci devra concéder que le peuple égyptien est le plus grand de tous les peuples...",]
X_new = vectorizer.transform(new_videos)
preds = clf.predict(X_new)
predicted_tags = mlb.inverse_transform(preds)

print("New video:", new_videos)
print("Suggested tags:", predicted_tags)