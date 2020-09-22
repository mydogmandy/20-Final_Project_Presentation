-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "Vehicles" (
    "index" int   NOT NULL,
    "yearVeh" int   NOT NULL,
    "airbag" varchar   NOT NULL,
    "abcat" varchar   NOT NULL,
    CONSTRAINT "pk_Vehicles" PRIMARY KEY (
        "index"
     )
);

CREATE TABLE "Accident" (
    "index" int   NOT NULL,
    "dvcat" varchar   NOT NULL,
    "frontal" int   NOT NULL,
    "caseid" varchar   NOT NULL,
    CONSTRAINT "pk_Accident" PRIMARY KEY (
        "index"
     )
);

CREATE TABLE "Occupant" (
    "index" int   NOT NULL,
    "dead" varchar   NOT NULL,
    "seatbelt" varchar   NOT NULL,
    "sex" varchar   NOT NULL,
    "ageOFocc" int   NOT NULL,
    "occRole" varchar   NOT NULL,
    "deploy" int   NOT NULL,
    "injSeverity" int   NOT NULL,
    CONSTRAINT "pk_Occupant" PRIMARY KEY (
        "index"
     )
);

ALTER TABLE "Vehicles" ADD CONSTRAINT "fk_Vehicles_index" FOREIGN KEY("index")
REFERENCES "Occupant" ("index");

ALTER TABLE "Accident" ADD CONSTRAINT "fk_Accident_index" FOREIGN KEY("index")
REFERENCES "Occupant" ("index");

