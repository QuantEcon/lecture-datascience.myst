filename <- "ahs2011forjep.rdata"
if (!file.exists(filename)) {
  # AHS data as used and cleaned by from Mullainathan & Spiess (2017)

  ## Download and run code used by Mullainathan & Spiess (2017)
  if (!file.exists("4211.zip")) {
    download.file("https://www.aeaweb.org/articles/attachments?retrieve=hfL4yDm2epidrc9MWUOkEPFSp-mJmvum",
                  destfile="4211.zip")
  }
  if (!file.exists("MullainathanSpiess/0_dataprep.R")) {
    unzip("4211.zip")
    ## why does MAC's zip create useless __MACOSX directory and  files?
    unlink("__MACOSX", recursive=TRUE) ## delete that garbage
  }

  if (!file.exists("ahs2011forjep.rdata")) {
    # download data
    if (!file.exists("tnewhouse.zip")) {
      download.file("http://www2.census.gov/programs-surveys/ahs/2011/AHS%202011%20National%20and%20Metropolitan%20PUF%20v1.4%20CSV.zip",
                    destfile="tnewhouse.zip")
    }
    unzip("tnewhouse.zip",files="tnewhouse.csv")

    # run M&S dataprep code
    source("MullainathanSpiess/0_dataprep.R")
  }
}

ahs <- readRDS(filename)
df <- ahs[["df"]]
df <- df[,!(names(df) %in% c("holdout","lassofolds"))]
df <- df[,!(grepl("MISS$",names(df)))]
set.seed(11032019)
ntrain  <- 2500
traini <- sample.int(nrow(ahs[["df"]]), ntrain)
traindf <- df[traini,]
evaldf  <- df[-traini,]

write.csv(traindf,"ahs-train.csv", row.names=FALSE)
write.csv(evaldf, "ahs-eval.csv", row.names=FALSE)


doc <- read.csv("AHSDICT_11MAR19_10_52_04_41_S.csv")
doc <- doc[doc$Variable %in% names(traindf),]
write.csv("ahs-doc.csv",row.names=FALSE)
