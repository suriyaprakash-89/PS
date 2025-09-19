import React from 'react';
import { useParams, Navigate } from 'react-router-dom';

// Import all your specific exam page components
import ML_ExamPage from './ML_ExamPage';
import DS_ExamPage from './DS_ExamPage';
import SpeechRecognition_ExamPage from './SpeechRecognition_ExamPage'; // Ensure this path is correct

const ExamPageDispatcher = () => {
  const { subject } = useParams();

  // --- THIS IS THE FIX ---
  // We normalize the subject key to make the matching reliable.
  // This removes all spaces and converts to lowercase.
  // "Speech Recognition" becomes "speechrecognition"
  const normalizedSubject = subject?.replace(/\s+/g, '').toLowerCase();

  // Use a switch statement on the NORMALIZED subject key
  switch (normalizedSubject) {
    case 'ml':
      return <ML_ExamPage />;
    
    case 'ds':
      return <DS_ExamPage />;

    case 'speechrecognition': // <-- Match the new, normalized key
      return <SpeechRecognition_ExamPage />;

    // You can add cases for your other subjects here using their normalized keys
    // Example: case 'generativeai': return <GenAI_ExamPage />;
    // Example: case 'deepllearning': return <DL_ExamPage />;

    default:
      // This log helps debug any future issues
      console.error(`No exam component found for subject: "${subject}" (Normalized to: "${normalizedSubject}")`);
      return <Navigate to="/dashboard" />;
  }
};

export default ExamPageDispatcher;