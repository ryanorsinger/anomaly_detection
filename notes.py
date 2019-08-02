# RESEARCH QUESTIONS
# How much are webdev folks are accessing data science curriculum?
# How much are data science folks accessing webdev curriculum?
# How much are people accessing the curriculum after graduating?
# Are there users who aren't accessing the curriculum very much at all? 
# What's the longest period of time someone's accessed the curriculum: ex. Started class in Jan 2018 and keeps accessing it?
# What outliers are there in IP addresses?
# What outliers are there in individuals accessing the curriculum?
# What's the most commonly accessed topic module?
# What's the least accessed topic module?

# Set timestamp as the index
# Add columns: cohort name, cohort_type

courses = [
    "Web Development",
    "Data Science"
]

role = [
    "Staff",
    "Instructor",
    "Student",
    "Unknown"
]

web_dev_modules = [
    "HTML/CSS",
    "JS I",
    "jQuery",
    "JS II",
    "JavaScript II",
    "Java I",
    "Java II",
    "MySQL",
    "Java III",
    "Spring",
    "Appendix"
]

data_science_modules = [
    "Foundations",
    "Tools",
    "Methodologies",
    "Appendix"
]

df.info()