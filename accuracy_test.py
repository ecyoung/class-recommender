"""
Author: Eric Young, Jannis Stoeter
File Name: accuracy_test.py
Purpose: calculates a series of tests (RSME) to analyze the accuracy of our recommender system
Approach:
Split course ratings data into two sets: training and test. 
Predict ratings for the test data only using data from the training data. 
“Average” error but penalizes bad large mistakes more and small mistakes less than mean absolute error.
"""

import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from input_parsing import parseInputData
import collaborative_filtering
import content_filtering_optimization_script
import content_filtering
import knn_util

"""
Calculate the average of all ratings 
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :return: average for all user ratings
"""
def getAvgRating(users):
    userList = users.keys()
    # sum of all ratings
    ratingSum = 0
    # total number of ratings 
    ratingCount = 0
    for user_id in userList:
        # user is a User object that contains information about preferences and ratings 
        user = users[user_id]
        # increment ratingSum by all ratings of the user and update total count
        for rating in user.courseRatings.values():
            ratingSum += rating
            ratingCount += 1
    avgRating = ratingSum / ratingCount
    return avgRating


"""--------- accuracy tests for a single user -----------"""

"""
Calculates the RMSE of predicted course ratings vs actual course ratings for one user
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param user_id: user_id of the user we want to calculate RMSE
    :param k: k nearest neighbor which we based our recommendation on
    :param comparatorSet: set of attributes to compare
    :return: RMSE value for user's predicted vs actual course ratings
"""

def getRMSE(users, user_id, k, comparatorSet):
        # user is a User object that contains information about preferences and ratings
        user = users[user_id]
        # total number of course ratings for the user
        numRatings = len(user.courseRatings)
        # getPredictedCourseRatings(users, user_id, k, threshold, comparatorSet, r, rho, is_validation, pref_weight=0.5, course_weight=0.5)
        predicted_ratings = collaborative_filtering.getPredictedCourseRatings(users, user_id, k, 0, comparatorSet, 70, 3, True)
        sumOfSquaredDiffs = 0
        for course_id, rating in predicted_ratings:
            actualRating = user.courseRatings[course_id]
            predictedRating = rating
            squaredDiff = (actualRating - predictedRating)**2
            # increment sumOfSquaredDiffs by the squaredDiff of current predicted vs actual rating
            sumOfSquaredDiffs += squaredDiff
        # divide sumOfSquaredDiffs by numRatings to obtain average and take square root
        rmse = math.sqrt(sumOfSquaredDiffs / numRatings)
        return rmse

"""
Calculates the baseline RMSE for one user
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param user_id: user_id of the user we want to calculate RMSE
    :param defaultRating: default predicted rating
    :return: baseline RMSE
"""
def getBaselineRMSE(users, user_id, courses):
    # get user object
    user = users[user_id]
    courseRatings = user.courseRatings
    numRatings = len(courseRatings)
    sumOfSquaredDiffs = 0
    for course in courseRatings.keys():
        squaredDiff = (courseRatings[course] - courses[course].average_rating) ** 2
        sumOfSquaredDiffs += squaredDiff
    # divide sumOfSquaredDiffs by numRatings to obtain average and take square root
    baselineRMSE = math.sqrt(sumOfSquaredDiffs / numRatings)
    return baselineRMSE


"""--------- accuracy tests across all users -----------"""

"""
Calculates the RMSE across all users by calculating the error term for each possible (user, course) pair and add them altogether inside the square root
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param k: k nearest neighbor which we based our recommendation on
    :param comparatorSet: set of attributes to compare
    :param pref_weight: weight of preference similarity between users to determine overall similarity
    :param course_weight: weight of course similarity between users to determine overall similarity
    :return: RMSE value for all users
"""
def getTotalRMSE(users, courses, k, comparatorSet, pref_weight = 0.5, course_weight = 0.5, option=1, type_of_class_weight=.1, topic_weight=0.15, knn_weight=.75):
    userList = users.keys()
    # initialize total terms
    totalTerms = 0
    # initialize sum of squared diffs term
    sumOfSquaredDiffs = 0
    for user_id in userList:
        # user is a User object that contains information about preferences and ratings 
        user = users[user_id]
        # total number of course ratings for the user
        numRatings = len(user.courseRatings)
        if option == 2:
            predicted_ratings = content_filtering.getPredictedCourseRatingsApproach1(courses, users, user_id,
                                                                                     comparatorSet, True, 80, 1, k,
                                                                                     pref_weight, course_weight,
                                                                                     type_of_class_weight, topic_weight,
                                                                                     knn_weight)
        else:
            predicted_ratings = collaborative_filtering.getPredictedCourseRatings(users, courses, user_id,
                                                                                  comparatorSet, True, 80, 1, k,
                                                                                  pref_weight, course_weight)
        courses_to_rate = set(users[user_id].courseRatings.keys())

        for course_id, rating in predicted_ratings:
            if course_id in courses_to_rate:
                courses_to_rate.remove(course_id)
            actualRating = user.courseRatings[course_id]
            predictedRating = rating
            squaredDiff = (actualRating - predictedRating) ** 2
            # increment sumOfSquaredDiffs by the squaredDiff of current predicted vs actual rating
            sumOfSquaredDiffs += squaredDiff
        # increment total terms with number of ratings of user
        for course_id in courses_to_rate:
            actualRating = user.courseRatings[course_id]
            predictedRating = courses[course_id].average_rating
            squaredDiff = (actualRating - predictedRating) ** 2
            sumOfSquaredDiffs += squaredDiff
        # increment total terms with number of ratings of user
        totalTerms += numRatings
    # divide sumOfSquaredDiffs by total terms to obtain average and take square root 
    rmse = math.sqrt(sumOfSquaredDiffs / totalTerms)
    return rmse

"""
Calculates the RMSE as above, but uses different parameters for runtime improvements during optimization
"""

def getTotalRMSE_opt(kNN, users, courses, k):
    userList = users.keys()
    # initialize total terms
    totalTerms = 0
    # initialize sum of squared diffs term
    sumOfSquaredDiffs = 0
    for user_id in userList:
        # user is a User object that contains information about preferences and ratings
        user = users[user_id]
        # total number of course ratings for the user
        numRatings = len(user.courseRatings)
        predicted_ratings = collaborative_filtering.getPredictedCourseRatings_opt((kNN[user_id])[:k], users, user_id, courses, True)

        courses_to_rate = set(users[user_id].courseRatings.keys())

        for course_id, rating in predicted_ratings:
            if course_id in courses_to_rate:
                courses_to_rate.remove(course_id)
            actualRating = user.courseRatings[course_id]
            predictedRating = rating
            squaredDiff = (actualRating - predictedRating)**2
            # increment sumOfSquaredDiffs by the squaredDiff of current predicted vs actual rating
            sumOfSquaredDiffs += squaredDiff
        # increment total terms with number of ratings of user
        for course_id in courses_to_rate:
            actualRating = user.courseRatings[course_id]
            predictedRating = courses[course_id].average_rating
            squaredDiff = (actualRating - predictedRating) ** 2
            sumOfSquaredDiffs += squaredDiff

        totalTerms += numRatings
    # divide sumOfSquaredDiffs by total terms to obtain average and take square root
    rmse = math.sqrt(sumOfSquaredDiffs / totalTerms)
    return rmse


"""
Calculates the total baseline RMSE across all users by calculating the error term for each possible (user, course) pair and add them altogether inside the square root
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param defaultRating: default predicted rating
    :return: Average baseline RMSE value for all users
"""
def getTotalBaselineRMSE(users, courses):
    userList = users.keys()
    # initialize total terms
    totalTerms = 0
    # initialize sum of squared diffs term
    sumOfSquaredDiffs = 0
    for user_id in userList:
        # get user object
        user = users[user_id]
        courseRatings = user.courseRatings
        numRatings = len(courseRatings)
        for course in courseRatings.keys():
            squaredDiff = (courseRatings[course] - courses[course].average_rating) ** 2
            sumOfSquaredDiffs += squaredDiff
        # increment total terms with number of ratings of user
        totalTerms += numRatings
    # divide sumOfSquaredDiffs by total terms to obtain average and take square root 
    baselineRMSE = math.sqrt(sumOfSquaredDiffs / totalTerms)
    return baselineRMSE


""" ------------------- average values ------------------------- """

"""
Calculates the average RMSE across all users
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param defaultRating: default predicted rating
    :return: Average  RMSE value for all users
"""
def getAverageRMSE(users, k, comparatorSet):
    userList = users.keys()
    numUsers = len(userList)
    sumOfBaselineRMSE = 0
    for user_id in userList:
        userBaselineRMSE = getRMSE(users, user_id, k, comparatorSet)
        sumOfBaselineRMSE += userBaselineRMSE
    averageBaselineRMSE = sumOfBaselineRMSE/numUsers
    return averageBaselineRMSE

"""
Calculates the average baseline RMSE across all users
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param courses: dictionary that maps course_id to Course object, which contains average ratings for each course
    :return: Average baseline RMSE value for all users
"""
def getAverageBaselineRMSE(users, courses):
    userList = users.keys()
    numUsers = len(userList)
    sumOfBaselineRMSE = 0
    for user_id in userList:
        userBaselineRMSE = getBaselineRMSE(users, user_id, courses)
        sumOfBaselineRMSE += userBaselineRMSE
    averageBaselineRMSE = sumOfBaselineRMSE/numUsers
    return averageBaselineRMSE


""" ---------- optimization script --------------"""

"""
Calculates the optimal k value in combinations with course and preference weights that collectively minimizes RMSE value 
    :param users: dictionary that maps user_id to User object, which contains information about preferences and courseRatings
    :param comparatorSet: set of attributes to compare
    :return: Most optimal k value in combination with preference/course weights with associated RMSE value
"""
def getOptimalSettings(users, courses):
    # increments/decrements of 0.05
    userList = users.keys()
    kNN = dict()

    best_combo = None
    minRMSE = float("inf")
    weightCombinations = [(0,1), (0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.4, 0.6), (0.45, 0.55), (0.5, 0.5), (0.55, 0.45), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.85, 0.15), (0.9, 0.1), (0.95, 0.05), (1,0)]

    for pref_weight, course_weight in weightCombinations:
        for user_id in userList:
            kNN[user_id] = knn_util.knn_opt(users, user_id, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, pref_weight, course_weight)
        for k in range(1, 50):
            RMSE = getTotalRMSE_opt(kNN, users, courses, k)
            if RMSE < minRMSE:
                minRMSE = RMSE
                best_combo = (k, pref_weight, course_weight)
    print("")
    print("Minimized RSME: " + str(minRMSE))
    print("Best_combo: " + str(best_combo))
    return best_combo


# main function for accuracy testing
if __name__ == "__main__":
    # create the user dictionary that maps user_id (int) to the User object
    users = {}
    # create the course dictionary that maps course_id (int) to the Course object (id, name)
    courses = {}
    # process users and courses dictionaries
    (users, courses) = parseInputData("survey_responses.csv", "course_tags.csv")
    avgRating = getAvgRating(users)
    print("Average Rating:", avgRating)
    # utilizes all 18 surveyed preference attributes
    comparatorSet = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    print("Baseline RMSE: ", getTotalBaselineRMSE(users, courses))
    print("RMSE Collaborative Filtering: " + str(getTotalRMSE(users, courses, 20, comparatorSet, 0.2, 0.8)))
    print("RMSE Content Filtering Approach 1: " + str(getTotalRMSE(users, courses, 20, comparatorSet, 0.2, 0.8, 2)))

    # getOptimalSettings(users, courses)
    # result = content_filtering_optimization_script.findOptimalValuesApproach1(courses, users)
    # print("")
    # print("Optimal settings: ")
    # print("k: " + str((result[0])[0]))
    # print("pref_weight: " + str((result[0])[1]))
    # print("course_weight: " + str((result[0])[2]))
    # print("knn_weight: " + str((result[0])[3]))
    # print("type_of_class_weight: " + str((result[0])[4]))
    # print("topic_weight: " + str((result[0])[5]))
    # print("RMSE: " + str(result[1]))