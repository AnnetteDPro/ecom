"""

Config file for Streamlit App

"""

[remote "origin"]
    url = git@bitbucket.org:AnnetteDPro/ecom.git/streamlit_app.git
    fetch = +refs/heads/*:refs/remotes/origin/*
        
from member import Member


TITLE = "E-COM"

TEAM_MEMBERS = [
    Member(
        name="Annette Dubus",
        linkedin_url="https://www.linkedin.com/in/annadubus",
        github_url="https://github.com/charlessutton",
    ),
    Member("TO Thi Phuong Thao"),
    Member("Camille LARA"),
]

PROMOTION = "Cohorte Data-Analystes février 2022 de DataScientest"
