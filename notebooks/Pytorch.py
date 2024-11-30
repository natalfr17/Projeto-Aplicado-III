import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Criar o dataset
class BookLensDataset(Dataset):
    def __init__(self, users, books, ratings):
        self.users = users
        self.books = books
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        return {
            "users": torch.tensor(self.users[item], dtype=torch.long),
            "books": torch.tensor(self.books[item], dtype=torch.long),
            "ratings": torch.tensor(self.ratings[item], dtype=torch.float),
        }

# Modelo de recomendação
class RecommendationSystemModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_size=256, hidden_dim=256, dropout_rate=0.2):
        super(RecommendationSystemModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc1 = nn.Linear(2 * embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, users, books):
        user_embedded = self.user_embedding(users)  # (batch_size, embedding_size)
        book_embedded = self.book_embedding(books)  # (batch_size, embedding_size)
        combined = torch.cat([user_embedded, book_embedded], dim=1)  # (batch_size, 2 * embedding_size)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)  # (batch_size, 1)
        return output


# Carregar dados
chunks = pd.read_json('goodreads_reviews_dedup.json', lines=True, chunksize=1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for c in chunks:
    # print(c.info())

    # Dados fictícios para teste
    users = c['user_id'].factorize()[0]
    books = c['book_id'].factorize()[0]
    ratings = c['rating']

    dataset = BookLensDataset(users, books, ratings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instanciar o modelo
    num_users = len(set(users))
    num_books = len(set(books))
    model = RecommendationSystemModel(num_users, num_books)

    # Testar com um lote do DataLoader
    for batch in dataloader:
        users = batch["users"]
        books = batch["books"]
        ratings = batch["ratings"]
        output = model(users, books)
        print("Output shape:", output.shape)  # Deve ser [batch_size, 1]
        break


    from sklearn import preprocessing
    from sklearn import model_selection
    from torch.utils.data import DataLoader

    # Preparação dos dados com LabelEncoder
    le_user = preprocessing.LabelEncoder()
    le_book = preprocessing.LabelEncoder()
    c.user_id = le_user.fit_transform(c.user_id.values)
    c.book_id = le_book.fit_transform(c.book_id.values)

    # Divisão em conjuntos de treinamento e validação
    df_train, df_val = model_selection.train_test_split(
        c, test_size=0.1, random_state=3, stratify=c.rating.values
    )

    # Criação de instâncias dos datasets
    train_dataset = BookLensDataset(
        df_train.user_id.values, df_train.book_id.values, df_train.rating.values
    )
    valid_dataset = BookLensDataset(
        df_val.user_id.values, df_val.book_id.values, df_val.rating.values
    )

    # Configuração dos DataLoaders
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Configuração do modelo
    recommendation_model = RecommendationSystemModel(
        num_users=len(le_user.classes_), 
        num_books=len(le_book.classes_),
        embedding_size=128,
        hidden_dim=256,
        dropout_rate=0.1,
    ).to(device)

    # Configuração do otimizador e função de perda
    optimizer = torch.optim.Adam(recommendation_model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    # Loop de treinamento
    EPOCHS = 2
    for epoch in range(EPOCHS):
        recommendation_model.train()
        total_loss = 0
        for batch in train_loader:
            users = batch["users"].to(device)
            books = batch["books"].to(device)
            ratings = batch["ratings"].to(torch.float32).to(device)

            optimizer.zero_grad()
            outputs = recommendation_model(users, books).squeeze()
            loss = loss_func(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    break
