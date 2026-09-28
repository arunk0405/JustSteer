import React from 'react';
import { Link } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import { Button } from '../components/ui/Button';
import { Footer } from '../components/layout/Footer';
import { HeroRouteVisual } from '../components/landing/HeroRouteVisual';
import styles from './LandingPage.module.css';

export const LandingPage: React.FC = () => {
  return (
    <>
      <PageContainer className={styles.container}>
        
        {/* HERO SECTION */}
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <div className={styles.eyebrow}>AI-POWERED ROUTE OPTIMIZATION</div>
            <h1 className={styles.headline}>
              Smarter Routes.<br />
              Safer Deliveries.
            </h1>
            <p className={styles.subtitle}>
              Plan multi-stop deliveries using traffic-aware and safety-aware route optimization powered by machine learning.
            </p>
            <div className={styles.actions}>
              <Link to="/plan" tabIndex={-1}>
                <Button size="lg" variant="primary">
                  Plan Your Route &rarr;
                </Button>
              </Link>
              <Link to="/how-it-works" tabIndex={-1}>
                <Button size="lg" variant="secondary">
                  How It Works
                </Button>
              </Link>
            </div>
          </div>
          
          <div className={styles.heroVisual}>
            <HeroRouteVisual />
          </div>
        </section>

        {/* VALUE PROPOSITION STRIP */}
        <section className={styles.featuresStrip}>
          <div className={styles.featureItem}>
            <div className={styles.featureIcon}>🔄</div>
            <h3 className={styles.featureTitle}>Optimized Sequence</h3>
            <p className={styles.featureDesc}>Find an efficient order for multiple delivery stops.</p>
          </div>
          <div className={styles.featureItem}>
            <div className={styles.featureIcon}>🧠</div>
            <h3 className={styles.featureTitle}>ML Predictions</h3>
            <p className={styles.featureDesc}>Adapt routing using predicted traffic and safety conditions.</p>
          </div>
          <div className={styles.featureItem}>
            <div className={styles.featureIcon}>⚡</div>
            <h3 className={styles.featureTitle}>Dynamic Routing</h3>
            <p className={styles.featureDesc}>Calculate route costs using changing road conditions.</p>
          </div>
          <div className={styles.featureItem}>
            <div className={styles.featureIcon}>📊</div>
            <h3 className={styles.featureTitle}>Explainable Results</h3>
            <p className={styles.featureDesc}>Understand how the route was calculated.</p>
          </div>
        </section>

        {/* WHY JUSTSTEER? */}
        <section className={styles.whySection}>
          <div className={styles.whyHeader}>
            <h2 className={styles.sectionTitle}>Route planning is more than finding the shortest road.</h2>
            <p className={styles.sectionSubtitle}>
              Traditional routing can focus primarily on distance. JustSteer combines traffic predictions, safety signals, and multi-stop optimization to build a route that considers more than distance alone.
            </p>
          </div>
          
          <div className={styles.comparisonGrid}>
            <div className={styles.comparisonCard}>
              <h3 className={styles.comparisonTitle}>Traditional Routing</h3>
              <ul className={styles.comparisonList}>
                <li>Shortest distance</li>
                <li>Static assumptions</li>
                <li>Basic stop ordering</li>
              </ul>
            </div>
            
            <div className={`${styles.comparisonCard} ${styles.comparisonHighlighted}`}>
              <h3 className={styles.comparisonTitle}>JustSteer</h3>
              <ul className={styles.comparisonList}>
                <li>Traffic-aware</li>
                <li>Safety-aware</li>
                <li>Optimized multi-stop sequence</li>
                <li>Explainable decisions</li>
              </ul>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS PREVIEW */}
        <section className={styles.howItWorksPreview}>
          <div className={styles.howItWorksHeader}>
            <h2 className={styles.sectionTitle}>From delivery locations to an optimized route.</h2>
          </div>
          
          <div className={styles.stepsGrid}>
            <div className={styles.stepItem}>
              <div className={styles.stepNumber}>01</div>
              <h3 className={styles.stepTitle}>Enter Stops</h3>
            </div>
            <div className={styles.stepItem}>
              <div className={styles.stepNumber}>02</div>
              <h3 className={styles.stepTitle}>Predict Traffic & Safety</h3>
            </div>
            <div className={styles.stepItem}>
              <div className={styles.stepNumber}>03</div>
              <h3 className={styles.stepTitle}>Optimize the Route</h3>
            </div>
            <div className={styles.stepItem}>
              <div className={styles.stepNumber}>04</div>
              <h3 className={styles.stepTitle}>Visualize & Explain</h3>
            </div>
          </div>
          
          <div className={styles.howItWorksAction}>
            <Link to="/how-it-works" tabIndex={-1}>
              <Button variant="ghost">Explore How It Works &rarr;</Button>
            </Link>
          </div>
        </section>

        {/* ML / ROUTING VISUAL */}
        <section className={styles.pipelineVisual}>
          <div className={styles.pipelineItems}>
            <div className={styles.pipelineNode}>Delivery Locations</div>
            <div className={styles.pipelineArrow}>↓</div>
            <div className={styles.pipelineNode}>Traffic + Safety Predictions</div>
            <div className={styles.pipelineArrow}>↓</div>
            <div className={styles.pipelineNode}>Dynamic Edge Costs</div>
            <div className={styles.pipelineArrow}>↓</div>
            <div className={styles.pipelineNode}>A* Routing</div>
            <div className={styles.pipelineArrow}>↓</div>
            <div className={styles.pipelineNode}>Multi-Stop Optimization</div>
            <div className={styles.pipelineArrow}>↓</div>
            <div className={styles.pipelineNode}>Explainable Route</div>
          </div>
        </section>

        {/* FINAL CTA */}
        <section className={styles.finalCta}>
          <h2 className={styles.ctaTitle}>Ready to plan a smarter route?</h2>
          <p className={styles.ctaSubtitle}>
            Enter your starting point and delivery locations to see how JustSteer builds an optimized route.
          </p>
          <Link to="/plan" tabIndex={-1}>
            <Button size="lg" variant="primary">
              Plan Your Route &rarr;
            </Button>
          </Link>
        </section>

      </PageContainer>
      
      <Footer />
    </>
  );
};
