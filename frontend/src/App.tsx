import { Routes, Route } from 'react-router-dom';
import { Header } from './components/layout/Header';
import { LandingPage } from './pages/LandingPage';
import { PlanRoutePage } from './pages/PlanRoutePage';
import { HowItWorksPage } from './pages/HowItWorksPage';

export default function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/plan" element={<PlanRoutePage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        {/* /results will be added later when integration happens */}
      </Routes>
    </>
  );
}
