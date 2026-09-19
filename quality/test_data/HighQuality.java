package com.example.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Represents an immutable, sorted collection of unique student records.
 * Provides efficient lookup by ID and grade-based filtering.
 *
 * <p>Thread-safe: all fields are final and the internal list is unmodifiable.</p>
 */
public class StudentRegistry {

    /** Internal sorted list of student records. */
    private final List<Student> students;

    /**
     * Creates a registry from a list of students, removing duplicates.
     *
     * @param rawStudents the input list; may contain duplicates
     * @throws NullPointerException if rawStudents is null
     */
    public StudentRegistry(List<Student> rawStudents) {
        Objects.requireNonNull(rawStudents, "Student list must not be null");
        List<Student> unique = new ArrayList<>();
        for (Student student : rawStudents) {
            if (!containsId(unique, student.getId())) {
                unique.add(student);
            }
        }
        Collections.sort(unique);
        this.students = Collections.unmodifiableList(unique);
    }

    /**
     * Returns the number of registered students.
     *
     * @return size of the registry
     */
    public int size() {
        return students.size();
    }

    /**
     * Finds a student by their unique identifier.
     *
     * @param studentId the ID to search for
     * @return the matching Student, or null if not found
     */
    public Student findById(int studentId) {
        for (Student student : students) {
            if (student.getId() == studentId) {
                return student;
            }
        }
        return null;
    }

    /**
     * Returns all students whose grade is at or above the given threshold.
     *
     * @param minimumGrade the inclusive lower bound (0.0 - 4.0)
     * @return an unmodifiable list of qualifying students
     * @throws IllegalArgumentException if minimumGrade is out of range
     */
    public List<Student> filterByMinimumGrade(double minimumGrade) {
        if (minimumGrade < 0.0 || minimumGrade > 4.0) {
            throw new IllegalArgumentException(
                "Grade must be between 0.0 and 4.0, got: " + minimumGrade);
        }
        List<Student> result = new ArrayList<>();
        for (Student student : students) {
            if (student.getGrade() >= minimumGrade) {
                result.add(student);
            }
        }
        return Collections.unmodifiableList(result);
    }

    /**
     * Computes the average grade across all students.
     *
     * @return the mean grade, or 0.0 if the registry is empty
     */
    public double computeAverageGrade() {
        if (students.isEmpty()) {
            return 0.0;
        }
        double total = 0.0;
        for (Student student : students) {
            total += student.getGrade();
        }
        return total / students.size();
    }

    /**
     * Returns an unmodifiable view of all students.
     *
     * @return sorted list of students
     */
    public List<Student> getAllStudents() {
        return students;
    }

    private boolean containsId(List<Student> list, int id) {
        for (Student student : list) {
            if (student.getId() == id) {
                return true;
            }
        }
        return false;
    }
}
