/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dataset, ColumnInfo, DataType } from '../types';

// Helper to infer column types & generate column info
export function analyzeColumns(rawData: Record<string, any>[]): ColumnInfo[] {
  if (rawData.length === 0) return [];
  const keys = Object.keys(rawData[0]);
  
  return keys.map(key => {
    // Check values to infer type
    const nonNullValues = rawData.map(r => r[key]).filter(v => v !== undefined && v !== null && v !== '');
    const isBool = nonNullValues.every(v => typeof v === 'boolean' || v === 'Yes' || v === 'No' || v === 1 || v === 0) &&
                   nonNullValues.some(v => v === 'Yes' || v === 'No' || typeof v === 'boolean');
    
    let type: DataType = 'categorical';
    
    if (isBool) {
      type = 'boolean';
    } else {
      const allNumbers = nonNullValues.every(v => !isNaN(Number(v)));
      if (allNumbers && nonNullValues.length > 0) {
        type = 'numeric';
      }
    }

    const missingValues = rawData.filter(r => r[key] === undefined || r[key] === null || r[key] === '').length;
    const uniqueValues = new Set(nonNullValues).size;
    
    const info: ColumnInfo = {
      name: key,
      type,
      missingValues,
      uniqueValues
    };

    if (type === 'numeric') {
      const nums = nonNullValues.map(v => Number(v));
      if (nums.length > 0) {
        info.min = Math.min(...nums);
        info.max = Math.max(...nums);
        const sum = nums.reduce((a, b) => a + b, 0);
        info.mean = sum / nums.length;
        info.median = nums.sort((a, b) => a - b)[Math.floor(nums.length / 2)];
        
        const squareDiffs = nums.map(value => {
          const diff = value - (info.mean || 0);
          return diff * diff;
        });
        const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / nums.length;
        info.stdDev = Math.sqrt(avgSquareDiff);
      }
    }

    return info;
  });
}

// 1. Iris Dataset
const irisRaw: Record<string, any>[] = [
  { SepalLength: 5.1, SepalWidth: 3.5, PetalLength: 1.4, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 4.9, SepalWidth: 3.0, PetalLength: 1.4, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 4.7, SepalWidth: 3.2, PetalLength: 1.3, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 4.6, SepalWidth: 3.1, PetalLength: 1.5, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 5.0, SepalWidth: 3.6, PetalLength: 1.4, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 5.4, SepalWidth: 3.9, PetalLength: 1.7, PetalWidth: 0.4, Species: 'Setosa' },
  { SepalLength: 4.6, SepalWidth: 3.4, PetalLength: 1.4, PetalWidth: 0.3, Species: 'Setosa' },
  { SepalLength: 5.0, SepalWidth: 3.4, PetalLength: 1.5, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 4.4, SepalWidth: 2.9, PetalLength: 1.4, PetalWidth: 0.2, Species: 'Setosa' },
  { SepalLength: 4.9, SepalWidth: 3.1, PetalLength: 1.5, PetalWidth: 0.1, Species: 'Setosa' },
  { SepalLength: 7.0, SepalWidth: 3.2, PetalLength: 4.7, PetalWidth: 1.4, Species: 'Versicolor' },
  { SepalLength: 6.4, SepalWidth: 3.2, PetalLength: 4.5, PetalWidth: 1.5, Species: 'Versicolor' },
  { SepalLength: 6.9, SepalWidth: 3.1, PetalLength: 4.9, PetalWidth: 1.5, Species: 'Versicolor' },
  { SepalLength: 5.5, SepalWidth: 2.3, PetalLength: 4.0, PetalWidth: 1.3, Species: 'Versicolor' },
  { SepalLength: 6.5, SepalWidth: 2.8, PetalLength: 4.6, PetalWidth: 1.5, Species: 'Versicolor' },
  { SepalLength: 5.7, SepalWidth: 2.8, PetalLength: 4.5, PetalWidth: 1.3, Species: 'Versicolor' },
  { SepalLength: 6.3, SepalWidth: 3.3, PetalLength: 4.7, PetalWidth: 1.6, Species: 'Versicolor' },
  { SepalLength: 4.9, SepalWidth: 2.4, PetalLength: 3.3, PetalWidth: 1.0, Species: 'Versicolor' },
  { SepalLength: 6.6, SepalWidth: 2.9, PetalLength: 4.6, PetalWidth: 1.3, Species: 'Versicolor' },
  { SepalLength: 5.2, SepalWidth: 2.7, PetalLength: 3.9, PetalWidth: 1.4, Species: 'Versicolor' },
  { SepalLength: 6.3, SepalWidth: 3.3, PetalLength: 6.0, PetalWidth: 2.5, Species: 'Virginica' },
  { SepalLength: 5.8, SepalWidth: 2.7, PetalLength: 5.1, PetalWidth: 1.9, Species: 'Virginica' },
  { SepalLength: 7.1, SepalWidth: 3.0, PetalLength: 5.9, PetalWidth: 2.1, Species: 'Virginica' },
  { SepalLength: 6.3, SepalWidth: 2.9, PetalLength: 5.6, PetalWidth: 1.8, Species: 'Virginica' },
  { SepalLength: 6.5, SepalWidth: 3.0, PetalLength: 5.8, PetalWidth: 2.2, Species: 'Virginica' },
  { SepalLength: 7.6, SepalWidth: 3.0, PetalLength: 6.6, PetalWidth: 2.1, Species: 'Virginica' },
  { SepalLength: 4.9, SepalWidth: 2.5, PetalLength: 4.5, PetalWidth: 1.7, Species: 'Virginica' },
  { SepalLength: 7.3, SepalWidth: 2.9, PetalLength: 6.3, PetalWidth: 1.8, Species: 'Virginica' },
  { SepalLength: 6.7, SepalWidth: 2.5, PetalLength: 5.8, PetalWidth: 1.8, Species: 'Virginica' },
  { SepalLength: 7.2, SepalWidth: 3.6, PetalLength: 6.1, PetalWidth: 2.5, Species: 'Virginica' }
];

// 2. Titanic Dataset (with deliberate missing values in Age, Embarked to demonstrate data cleaning)
const titanicRaw: Record<string, any>[] = [
  { Pclass: 3, Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Fare: 7.25, Embarked: 'S', Survived: 0 },
  { Pclass: 1, Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Fare: 71.28, Embarked: 'C', Survived: 1 },
  { Pclass: 3, Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Fare: 7.92, Embarked: 'S', Survived: 1 },
  { Pclass: 1, Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Fare: 53.10, Embarked: 'S', Survived: 1 },
  { Pclass: 3, Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 8.46, Embarked: 'Q', Survived: 0 }, // Missing Age
  { Pclass: 1, Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Fare: 51.86, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'male', Age: 2, SibSp: 3, Parch: 1, Fare: 21.07, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'female', Age: 27, SibSp: 0, Parch: 2, Fare: 11.13, Embarked: 'S', Survived: 1 },
  { Pclass: 2, Sex: 'female', Age: 14, SibSp: 1, Parch: 0, Fare: 30.07, Embarked: 'C', Survived: 1 },
  { Pclass: 3, Sex: 'female', Age: 4, SibSp: 1, Parch: 1, Fare: 16.70, Embarked: 'S', Survived: 1 },
  { Pclass: 1, Sex: 'female', Age: 58, SibSp: 0, Parch: 0, Fare: 26.55, Embarked: 'S', Survived: 1 },
  { Pclass: 3, Sex: 'male', Age: 20, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'male', Age: 39, SibSp: 1, Parch: 5, Fare: 31.27, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'female', Age: 14, SibSp: 0, Parch: 0, Fare: 7.85, Embarked: null, Survived: 0 }, // Missing Embarked
  { Pclass: 2, Sex: 'female', Age: 55, SibSp: 0, Parch: 0, Fare: 16.00, Embarked: 'S', Survived: 1 },
  { Pclass: 3, Sex: 'male', Age: 2, SibSp: 4, Parch: 1, Fare: 29.13, Embarked: 'Q', Survived: 0 },
  { Pclass: 2, Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 13.00, Embarked: 'S', Survived: 1 }, // Missing Age
  { Pclass: 3, Sex: 'female', Age: 31, SibSp: 1, Parch: 0, Fare: 18.00, Embarked: 'S', Survived: 0 },
  { Pclass: 1, Sex: 'female', Age: 19, SibSp: 0, Parch: 2, Fare: 263.00, Embarked: 'S', Survived: 1 },
  { Pclass: 3, Sex: 'male', Age: 26, SibSp: 0, Parch: 0, Fare: 7.22, Embarked: 'C', Survived: 0 },
  { Pclass: 1, Sex: 'male', Age: 40, SibSp: 0, Parch: 0, Fare: 27.72, Embarked: 'C', Survived: 1 },
  { Pclass: 3, Sex: 'female', Age: 15, SibSp: 0, Parch: 0, Fare: 8.03, Embarked: 'Q', Survived: 1 },
  { Pclass: 2, Sex: 'male', Age: 34, SibSp: 0, Parch: 0, Fare: 13.00, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'female', Age: 8, SibSp: 3, Parch: 1, Fare: 21.07, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'female', Age: 38, SibSp: 1, Parch: 5, Fare: 31.39, Embarked: 'S', Survived: 1 },
  { Pclass: 3, Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 7.22, Embarked: 'C', Survived: 0 }, // Missing Age
  { Pclass: 1, Sex: 'male', Age: 19, SibSp: 3, Parch: 2, Fare: 263.00, Embarked: 'S', Survived: 0 },
  { Pclass: 3, Sex: 'female', Age: null, SibSp: 0, Parch: 0, Fare: 7.88, Embarked: 'Q', Survived: 1 }, // Missing Age
  { Pclass: 1, Sex: 'male', Age: 26, SibSp: 0, Parch: 0, Fare: 30.00, Embarked: 'C', Survived: 1 }
];

// 3. Diabetes Dataset
const diabetesRaw: Record<string, any>[] = [
  { Pregnancies: 6, Glucose: 148, BloodPressure: 72, SkinThickness: 35, Insulin: 0, BMI: 33.6, DiabetesPedigree: 0.627, Age: 50, Outcome: 1 },
  { Pregnancies: 1, Glucose: 85, BloodPressure: 66, SkinThickness: 29, Insulin: 0, BMI: 26.6, DiabetesPedigree: 0.351, Age: 31, Outcome: 0 },
  { Pregnancies: 8, Glucose: 183, BloodPressure: 64, SkinThickness: 0, Insulin: 0, BMI: 23.3, DiabetesPedigree: 0.672, Age: 32, Outcome: 1 },
  { Pregnancies: 1, Glucose: 89, BloodPressure: 66, SkinThickness: 23, Insulin: 94, BMI: 28.1, DiabetesPedigree: 0.167, Age: 21, Outcome: 0 },
  { Pregnancies: 0, Glucose: 137, BloodPressure: 40, SkinThickness: 35, Insulin: 168, BMI: 43.1, DiabetesPedigree: 2.288, Age: 33, Outcome: 1 },
  { Pregnancies: 5, Glucose: 116, BloodPressure: 74, SkinThickness: 0, Insulin: 0, BMI: 25.6, DiabetesPedigree: 0.201, Age: 30, Outcome: 0 },
  { Pregnancies: 3, Glucose: 78, BloodPressure: 50, SkinThickness: 32, Insulin: 88, BMI: 31.0, DiabetesPedigree: 0.248, Age: 26, Outcome: 1 },
  { Pregnancies: 10, Glucose: 115, BloodPressure: 0, SkinThickness: 0, Insulin: 0, BMI: 35.3, DiabetesPedigree: 0.134, Age: 29, Outcome: 0 },
  { Pregnancies: 2, Glucose: 197, BloodPressure: 70, SkinThickness: 45, Insulin: 543, BMI: 30.5, DiabetesPedigree: 0.158, Age: 53, Outcome: 1 },
  { Pregnancies: 8, Glucose: 125, BloodPressure: 96, SkinThickness: 0, Insulin: 0, BMI: 0.0, DiabetesPedigree: 0.232, Age: 54, Outcome: 1 },
  { Pregnancies: 4, Glucose: 110, BloodPressure: 92, SkinThickness: 0, Insulin: 0, BMI: 37.6, DiabetesPedigree: 0.191, Age: 30, Outcome: 0 },
  { Pregnancies: 10, Glucose: 168, BloodPressure: 74, SkinThickness: 0, Insulin: 0, BMI: 38.0, DiabetesPedigree: 0.537, Age: 34, Outcome: 1 },
  { Pregnancies: 10, Glucose: 139, BloodPressure: 80, SkinThickness: 0, Insulin: 0, BMI: 27.1, DiabetesPedigree: 1.441, Age: 57, Outcome: 0 },
  { Pregnancies: 1, Glucose: 189, BloodPressure: 60, SkinThickness: 23, Insulin: 846, BMI: 30.1, DiabetesPedigree: 0.398, Age: 59, Outcome: 1 },
  { Pregnancies: 5, Glucose: 166, BloodPressure: 72, SkinThickness: 19, Insulin: 175, BMI: 25.8, DiabetesPedigree: 0.587, Age: 51, Outcome: 1 },
  { Pregnancies: 7, Glucose: 100, BloodPressure: 0, SkinThickness: 0, Insulin: 0, BMI: 30.0, DiabetesPedigree: 0.484, Age: 32, Outcome: 1 },
  { Pregnancies: 0, Glucose: 118, BloodPressure: 84, SkinThickness: 47, Insulin: 230, BMI: 45.8, DiabetesPedigree: 0.551, Age: 31, Outcome: 1 },
  { Pregnancies: 7, Glucose: 107, BloodPressure: 74, SkinThickness: 0, Insulin: 0, BMI: 29.6, DiabetesPedigree: 0.254, Age: 31, Outcome: 1 },
  { Pregnancies: 1, Glucose: 103, BloodPressure: 30, SkinThickness: 38, Insulin: 83, BMI: 43.3, DiabetesPedigree: 0.183, Age: 33, Outcome: 0 },
  { Pregnancies: 1, Glucose: 115, BloodPressure: 70, SkinThickness: 30, Insulin: 96, BMI: 34.6, DiabetesPedigree: 0.529, Age: 32, Outcome: 1 },
  { Pregnancies: 3, Glucose: 126, BloodPressure: 88, SkinThickness: 41, Insulin: 235, BMI: 39.3, DiabetesPedigree: 0.704, Age: 27, Outcome: 0 },
  { Pregnancies: 8, Glucose: 99, BloodPressure: 84, SkinThickness: 0, Insulin: 0, BMI: 35.4, DiabetesPedigree: 0.388, Age: 50, Outcome: 0 },
  { Pregnancies: 7, Glucose: 196, BloodPressure: 90, SkinThickness: 0, Insulin: 0, BMI: 39.8, DiabetesPedigree: 0.451, Age: 41, Outcome: 1 },
  { Pregnancies: 9, Glucose: 119, BloodPressure: 80, SkinThickness: 35, MDInsulin: 0, BMI: 29.0, DiabetesPedigree: 0.263, Age: 29, Outcome: 1 },
  { Pregnancies: 11, Glucose: 143, BloodPressure: 94, SkinThickness: 33, Insulin: 146, BMI: 36.6, DiabetesPedigree: 0.254, Age: 51, Outcome: 1 },
  { Pregnancies: 10, Glucose: 125, BloodPressure: 70, SkinThickness: 26, Insulin: 115, BMI: 31.1, DiabetesPedigree: 0.205, Age: 41, Outcome: 1 },
  { Pregnancies: 7, Glucose: 147, BloodPressure: 76, SkinThickness: 0, Insulin: 0, BMI: 39.4, DiabetesPedigree: 0.257, Age: 43, Outcome: 1 },
  { Pregnancies: 1, Glucose: 97, BloodPressure: 66, SkinThickness: 15, Insulin: 140, BMI: 23.2, DiabetesPedigree: 0.487, Age: 22, Outcome: 0 },
  { Pregnancies: 13, Glucose: 145, BloodPressure: 82, SkinThickness: 19, Insulin: 110, BMI: 22.2, DiabetesPedigree: 0.245, Age: 57, Outcome: 0 },
  { Pregnancies: 5, Glucose: 117, BloodPressure: 92, SkinThickness: 0, Insulin: 0, BMI: 34.1, DiabetesPedigree: 0.337, Age: 38, Outcome: 0 }
];

// 4. Student Performance Dataset (Continuous target: PerformanceIndex, i.e. regression)
const studentPerformanceRaw: Record<string, any>[] = [
  { HoursStudied: 4.5, Attendance: 85, SleepHours: 7, Extracurricular: 'Yes', PerformanceIndex: 55 },
  { HoursStudied: 8.0, Attendance: 92, SleepHours: 6, Extracurricular: 'No', PerformanceIndex: 82 },
  { HoursStudied: 3.2, Attendance: 75, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 41 },
  { HoursStudied: 6.5, Attendance: 88, SleepHours: 7, Extracurricular: 'Yes', PerformanceIndex: 71 },
  { HoursStudied: 9.1, Attendance: 96, SleepHours: 5, Extracurricular: 'Yes', PerformanceIndex: 94 },
  { HoursStudied: 2.0, Attendance: 60, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 25 },
  { HoursStudied: 5.5, Attendance: 80, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 60 },
  { HoursStudied: 7.2, Attendance: 90, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 78 },
  { HoursStudied: 3.8, Attendance: 78, SleepHours: 9, Extracurricular: 'Yes', PerformanceIndex: 48 },
  { HoursStudied: 8.6, Attendance: 95, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 89 },
  { HoursStudied: 1.5, Attendance: 50, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 18 },
  { HoursStudied: 4.0, Attendance: 82, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 50 },
  { HoursStudied: 7.5, Attendance: 91, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 80 },
  { HoursStudied: 5.0, Attendance: 84, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 58 },
  { HoursStudied: 6.8, Attendance: 89, SleepHours: 7, Extracurricular: 'Yes', PerformanceIndex: 74 },
  { HoursStudied: 2.5, Attendance: 65, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 30 },
  { HoursStudied: 8.3, Attendance: 94, SleepHours: 6, Extracurricular: 'No', PerformanceIndex: 85 },
  { HoursStudied: 9.5, Attendance: 98, SleepHours: 5, Extracurricular: 'Yes', PerformanceIndex: 98 },
  { HoursStudied: 4.2, Attendance: 81, SleepHours: 8, Extracurricular: 'Yes', PerformanceIndex: 52 },
  { HoursStudied: 6.0, Attendance: 87, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 67 },
  { HoursStudied: 3.0, Attendance: 72, SleepHours: 9, Extracurricular: 'No', PerformanceIndex: 38 },
  { HoursStudied: 7.0, Attendance: 91, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 76 },
  { HoursStudied: 5.2, Attendance: 85, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 62 },
  { HoursStudied: 8.9, Attendance: 96, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 91 },
  { HoursStudied: 1.8, Attendance: 55, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 22 },
  { HoursStudied: 4.8, Attendance: 83, SleepHours: 7, Extracurricular: 'Yes', PerformanceIndex: 56 },
  { HoursStudied: 6.2, Attendance: 88, SleepHours: 7, Extracurricular: 'No', PerformanceIndex: 68 },
  { HoursStudied: 7.8, Attendance: 93, SleepHours: 6, Extracurricular: 'Yes', PerformanceIndex: 84 },
  { HoursStudied: 3.5, Attendance: 76, SleepHours: 8, Extracurricular: 'No', PerformanceIndex: 45 },
  { HoursStudied: 9.2, Attendance: 97, SleepHours: 5, Extracurricular: 'No', PerformanceIndex: 95 }
];

// 5. Employee Salary Dataset (Continuous target: Salary, i.e. regression)
const employeeSalaryRaw: Record<string, any>[] = [
  { Experience: 1.1, Education: 'Bachelor', Department: 'Marketing', Age: 23, Rating: 3, Salary: 42000 },
  { Experience: 1.5, Education: 'Bachelor', Department: 'Sales', Age: 25, Rating: 4, Salary: 46000 },
  { Experience: 2.0, Education: 'Master', Department: 'Engineering', Age: 26, Rating: 5, Salary: 65000 },
  { Experience: 2.9, Education: 'Bachelor', Department: 'Sales', Age: 28, Rating: 3, Salary: 54000 },
  { Experience: 3.5, Education: 'Bachelor', Department: 'HR', Age: 30, Rating: 4, Salary: 58000 },
  { Experience: 4.0, Education: 'PhD', Department: 'Engineering', Age: 32, Rating: 5, Salary: 85000 },
  { Experience: 4.5, Education: 'Bachelor', Department: 'Marketing', Age: 31, Rating: 2, Salary: 62000 },
  { Experience: 5.1, Education: 'Master', Department: 'Sales', Age: 33, Rating: 3, Salary: 72000 },
  { Experience: 5.9, Education: 'Master', Department: 'HR', Age: 35, Rating: 4, Salary: 75000 },
  { Experience: 6.0, Education: 'Bachelor', Department: 'Engineering', Age: 34, Rating: 5, Salary: 82000 },
  { Experience: 6.8, Education: 'PhD', Department: 'Engineering', Age: 37, Rating: 4, Salary: 98000 },
  { Experience: 7.1, Education: 'Master', Department: 'Marketing', Age: 36, Rating: 3, Salary: 85000 },
  { Experience: 8.2, Education: 'PhD', Department: 'HR', Age: 39, Rating: 5, Salary: 104000 },
  { Experience: 9.0, Education: 'Bachelor', Department: 'Sales', Age: 40, Rating: 4, Salary: 95000 },
  { Experience: 10.5, Education: 'Master', Department: 'Engineering', Age: 42, Rating: 5, Salary: 125000 },
  { Experience: 11.2, Education: 'PhD', Department: 'Marketing', Age: 45, Rating: 4, Salary: 132000 },
  { Experience: 12.0, Education: 'Bachelor', Department: 'HR', Age: 44, Rating: 3, Salary: 110000 },
  { Experience: 13.5, Education: 'PhD', Department: 'Engineering', Age: 47, Rating: 5, Salary: 156000 },
  { Experience: 14.0, Education: 'Master', Department: 'Sales', Age: 48, Rating: 4, Salary: 142000 },
  { Experience: 15.0, Education: 'Master', Department: 'Engineering', Age: 50, Rating: 5, Salary: 160000 },
  { Experience: 1.3, Education: 'Bachelor', Department: 'Engineering', Age: 24, Rating: 4, Salary: 52000 },
  { Experience: 3.1, Education: 'Master', Department: 'Sales', Age: 29, Rating: 3, Salary: 63000 },
  { Experience: 5.5, Education: 'PhD', Department: 'HR', Age: 34, Rating: 4, Salary: 88000 },
  { Experience: 7.5, Education: 'Bachelor', Department: 'Marketing', Age: 38, Rating: 5, Salary: 92000 },
  { Experience: 9.5, Education: 'Master', Department: 'Engineering', Age: 41, Rating: 4, Salary: 118000 },
  { Experience: 11.0, Education: 'PhD', Department: 'Sales', Age: 43, Rating: 5, Salary: 138000 },
  { Experience: 2.5, Education: 'Bachelor', Department: 'HR', Age: 27, Rating: 3, Salary: 51000 },
  { Experience: 4.8, Education: 'Master', Department: 'Marketing', Age: 32, Rating: 4, Salary: 74000 },
  { Experience: 6.2, Education: 'PhD', Department: 'Engineering', Age: 35, Rating: 5, Salary: 102000 },
  { Experience: 8.5, Education: 'Bachelor', Department: 'Sales', Age: 39, Rating: 4, Salary: 91000 }
];

export function getPresetDatasets(): Dataset[] {
  return [
    {
      name: 'Iris Flower Dataset',
      filename: 'iris.csv',
      rawData: irisRaw,
      columns: analyzeColumns(irisRaw),
      rowCount: irisRaw.length,
      colCount: Object.keys(irisRaw[0]).length
    },
    {
      name: 'Titanic Survival Dataset',
      filename: 'titanic.csv',
      rawData: titanicRaw,
      columns: analyzeColumns(titanicRaw),
      rowCount: titanicRaw.length,
      colCount: Object.keys(titanicRaw[0]).length
    },
    {
      name: 'Diabetes Wellness Dataset',
      filename: 'diabetes.csv',
      rawData: diabetesRaw,
      columns: analyzeColumns(diabetesRaw),
      rowCount: diabetesRaw.length,
      colCount: Object.keys(diabetesRaw[0]).length
    },
    {
      name: 'Student Performance Dataset',
      filename: 'student_performance.csv',
      rawData: studentPerformanceRaw,
      columns: analyzeColumns(studentPerformanceRaw),
      rowCount: studentPerformanceRaw.length,
      colCount: Object.keys(studentPerformanceRaw[0]).length
    },
    {
      name: 'Employee Salary Dataset',
      filename: 'employee_salary.csv',
      rawData: employeeSalaryRaw,
      columns: analyzeColumns(employeeSalaryRaw),
      rowCount: employeeSalaryRaw.length,
      colCount: Object.keys(employeeSalaryRaw[0]).length
    }
  ];
}
