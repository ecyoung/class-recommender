"""This file contains helper functions for knn determination"""

import math
import heapq

def getPreferenceDistance(user1, user2, comparatorSet):
    """
    Get the angular distance between two users based on their ratings on preferences of interest
    :param user1: User object of the first user
    :param user2: User object of the second user
    :param comparatorSet: Set of preference ids (int) corresponding to the preferences of interest
    :return: float, the distance between user 1 and user 2
    """
    # initialize variables
    dotProduct = 0  # numerator of angular distance
    squaredSum1 = 0 # sum of (user 1's rating)^2
    squaredSum2 = 0 # sum of (user 2's rating)^2
    # get the preference mapping ({preference_id: rating}) of the two users
    # unfiltered (contains irrelevant preferences not in the comparatorSet)
    preference1 = user1.preferences
    preference2 = user2.preferences
    # loop through preferences of interest and get dot product and squared sums
    for pref_id in comparatorSet:
        rating1 = preference1[pref_id]
        rating2 = preference2[pref_id]
        dotProduct += (rating1*rating2)
        squaredSum1 += (rating1**2)
        squaredSum2 += (rating2**2)
    # return angular distance
    denominator = math.sqrt(squaredSum1*squaredSum2)
    return 1 - (dotProduct/denominator)


def getCourseDistance(user1, user2, threshold):
    """
    Get the angular distance between two users based on their ratings on courses
    :param user1: User object of the first user
    :param user2: User object of the second user
    :param threshold: Commonality threshold of #overlapping courses, return 1 if threshold not reached
    :return: float, the angular distance between user 1 and user 2
    """
    # get the course rating mapping ({course_id: rating}) of the two users
    courseRatings1 = user1.courseRatings
    courseRatings2 = user2.courseRatings
    # find intersection (ids of courses that both users rate)
    commonCourseIds = set(courseRatings1.keys() & courseRatings2.keys())

    # return 1 if commonality threshold not reached
    if len(commonCourseIds) < threshold:
        return 1
    # initialize variables
    dotProduct = 0  # numerator of angular distance
    squaredSum1 = 0 # sum of (user 1's rating)^2
    squaredSum2 = 0 # sum of (user 2's rating)^2
    # loop through preferences of interest and get dot product and squared sums
    for course_id in commonCourseIds:
        rating1 = courseRatings1[course_id]
        rating2 = courseRatings2[course_id]
        dotProduct += (rating1*rating2)
        squaredSum1 += (rating1**2)
        squaredSum2 += (rating2**2)
    # return angular distance
    denominator = math.sqrt(squaredSum1*squaredSum2)
    # quick fix (since dot product numerator is also 0)
    if denominator == 0.0:
        return 0
    """
        Comment Jannis: why not 1 - dotProduct/denominator
        old code: return dotProduct/denominator
    """
    return 1 - dotProduct/denominator


def knn(users, user_id, threshold, k, comparatorSet, pref_weight=0.5, course_weight=0.5):
    """
    Get the K nearest neighbors of a user
    :param: users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    :param threshold: commonality threshold
    :param k: number of nearest neighbors
    :param comparatorSet: set of ids of target preferences
    :return: list of user_ids (int) of the K nearest neighbors
    """
    # initialize priority queue, which will store (distance, user_id) pair
    heap = []
    # if k is greater than the size of all users, return all user id
    if k > len(users):
        return users.keys()
    # compute the angular distances between the user and all other users
    for curr_id in users.keys():
        if user_id == curr_id:
            continue
        # get angular distances (takes in User object)
        preference_distance = getPreferenceDistance(users.get(user_id), users.get(curr_id), comparatorSet)
        course_distance = getCourseDistance(users.get(user_id), users.get(curr_id), threshold)
        distance = preference_distance*pref_weight + course_distance*course_weight
        # stores neighbor distances in the form of a tuple: (neighbor user_id, distance)
        heapq.heappush(heap, (distance, curr_id))

    knn = []
    for i in range(k):
        if len(heap) > 0:
            # append the user id to knn
            knn.append(heapq.heappop(heap)[1])
    return knn


def knn_opt(users, user_id, threshold, comparatorSet, pref_weight=0.5, course_weight=0.5):
    """
    Get the K nearest neighbors of a user
    :param: users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    :param threshold: commonality threshold
    :param k: number of nearest neighbors
    :param comparatorSet: set of ids of target preferences
    :return: list of user_ids (int) of the K nearest neighbors
    """
    # initialize priority queue, which will store (distance, user_id) pair
    heap = []

    # compute the angular distances between the user and all other users
    for curr_id in users.keys():
        if user_id == curr_id:
            continue
        # get angular distances (takes in User object)
        preference_distance = getPreferenceDistance(users.get(user_id), users.get(curr_id), comparatorSet)
        course_distance = getCourseDistance(users.get(user_id), users.get(curr_id), threshold)
        distance = preference_distance*pref_weight + course_distance*course_weight
        # stores neighbor distances in the form of a tuple: (neighbor user_id, distance)
        heapq.heappush(heap, (distance, curr_id))

    knn = []
    while len(heap) > 0:
        # append the user id to knn
        knn.append(heapq.heappop(heap)[1])

    return knn