from gain_imputer import GainImputerBuilder

cat_columns = [0, 1]
builder = (
    GainImputerBuilder()
    .with_dim(28)
    .with_h_dim(128)
    .with_cat_columns(cat_columns)
    .with_batch_size(2028)
    .build()
)
print(builder)
