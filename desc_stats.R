library(arsenal)
library(dplyr)

df <- read.csv("data/desc_stat.csv")
View(df)

table1 <- tableby(
    Cluster.label ~ .,
    data = df |> select(!NUM_ID)
)

table1 |>
summary(text=TRUE) |>
as.data.frame() |>
write.csv(file = "data/desc_table.csv", row.names=FALSE)
#View()
# TODO Survival proba (cf tableby vignette example) or incidence rate?
# TODO Median age instead of mean age
# TODO Specify units in a nice way