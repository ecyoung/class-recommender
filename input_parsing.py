"""This file contains helper functions for knn determination"""

import csv
import pandas as pd
from class_recommender import Course, User


def parseInputData(filepath_survey_data, filepath_tags):
    """
    Parses input data in CSV at filepath into a users and courses dictionary
    :return: (users, courses) where
        users is a dictionary that maps user_id to User object, contains information of all users
        courses is a dictionary that maps course_id to Course object, contains information of all courses
    """
    # create the user dictionary that maps user_id (int) to the User object
    courses = {}
    users = {}

    num_preferences = 18

    # reading csv file
    with open(filepath_survey_data, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        headers = next(csvreader)
        col_num = 0
        # iterate through all headers (minus first timestamp header)
        for header in headers[1:]:
            column_name = getColumnName(header)
            # course columns come after all preference columns (starting at column 18)
            if (col_num >= num_preferences):
                course_id = col_num
                course = Course(course_id, column_name)
                courses[col_num] = course
            col_num += 1

        # extracting each data row one by one
        row_num = 0
        for row in csvreader:
            user_id = row_num
            preference_ratings = {}
            course_ratings = {}

            col_num = 0
            # iterate through all columns (minus first timestamp column)
            for rating in row[1:]:
                if isValidRating(rating):
                    rate = getRating(rating)
                    # preference ratings are in columns: 0-17
                    if col_num < num_preferences:
                        preference_ratings[col_num] = rate
                    # class ratings come after preference ratings
                    else:
                        course_ratings[col_num] = rate
                        courses[col_num].ratings.append(rate)
                col_num += 1

            user = User(user_id, preference_ratings, course_ratings)
            users[user_id] = user
            row_num += 1

        for course in courses.keys():
            avg = 0
            ratings = courses[course].ratings
            n = len(ratings)
            if n == 0:
                courses[course].average_rating = 3.499
            else:
                for rating in ratings:
                    avg += rating/n
                courses[course].average_rating = (n*avg + 3.499)/(n+1)

    with open(filepath_tags, 'r') as tags:
        csvreader = csv.reader(tags)
        next(csvreader)

        row_num = 18
        for row in csvreader:
            course_id = row_num
            for tag in row:
                if 0 <= int(tag) <= 4:
                    courses[course_id].type_of_class_tags.append(int(tag))
                elif 13 <= int(tag) <= 17:
                    courses[course_id].topic_tags.append(int(tag))
            row_num += 1

    completed_users = getCompletedUsers(users, "complete_matrix.csv")
    return (completed_users, courses)

def getRating(rating_string):
    """
    Returns numerical rating (1-5) for string rating
    """
    ratings = {
        "Dislike": 1,
        "Hated": 1,
        "Somewhat Dislike": 2,
        "Disliked": 2,
        "Neutral": 3,
        "Liked": 4,
        "Somewhat Prefer": 4,
        "Loved": 5,
        "Prefer": 5
    }
    return ratings[rating_string]


def isValidRating(rating):
    """
    Returns whether the rating string is a valid rating or not. A rating is valid if it is not an empty string.
    """
    return len(rating) > 0


def getColumnName(header):
    """
    Returns the formatted column name from the CSV header.
    """
    return header.split("[")[1].split("]")[0]

def getCompletedUsers(users, matrixFile):
    df = pd.read_csv(matrixFile)
    completed_matrix = []
    for row in df.itertuples():
        ratings = []
        for rating in row:
            ratings.append(rating)
        completed_matrix.append(ratings[1:-1])

    completed_users = get_user_dictionary(completed_matrix, users)
    return completed_users

def get_user_dictionary(matrix, users):
    # course ratings
    for user_id in range(len(users)):
        for zero_indexed_course_id in range(len(matrix[user_id])):
            # course_id's range from [18: CS101 to 85: CS675D], so we need to offset zero_index_course_id by +18
            course_id = zero_indexed_course_id + 18
            users[user_id].factorizedCourseRatings[course_id] = matrix[user_id][zero_indexed_course_id]
    return users