class UserEmbedding(db.Model):
    __tablename__ = "user_embeddings"
    user_id = db.Column("user_id", db.BigInteger, primary_key=True)
    vector = db.Column("vector", db.Text, nullable=False)
    def to_array(self):
        return [float(x) for x in self.vector.split(",")]

class RecipeEmbedding(db.Model):
    __tablename__ = "recipe_embeddings"
    rcp_sno = db.Column("rcp_sno", db.BigInteger, primary_key=True)
    vector = db.Column("vector", db.Text, nullable=False)
    def to_array(self):
        return [float(x) for x in self.vector.split(",")]