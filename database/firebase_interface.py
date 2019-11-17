import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


class FirebaseInterface:

    def __init__(self):
        self.db = firestore.client()

    def addOrUpdateData(self, collection, document, data):
        doc_ref = self.db.collection(collection).document(document)
        try:
            doc_ref.update(data)
        except:
            doc_ref.set(data)
