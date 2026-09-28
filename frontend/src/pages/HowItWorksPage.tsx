import React from 'react';
import { PageContainer } from '../components/layout/PageContainer';
import { Card, CardTitle, CardContent } from '../components/ui/Card';
import { Footer } from '../components/layout/Footer';

export const HowItWorksPage: React.FC = () => {
  return (
    <>
      <PageContainer>
        <div style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
          <h1 style={{ marginBottom: '8px' }}>How JustSteer Works</h1>
          <p style={{ color: 'var(--color-text-secondary)', fontSize: '18px', marginBottom: '32px' }}>
            Understanding the intelligence behind the routing engine.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <Card>
              <CardContent>
                <CardTitle>1. Context Aggregation</CardTitle>
                <p style={{ color: 'var(--color-text-secondary)', marginTop: '8px' }}>
                  When you request a route, the system first gathers live weather and contextual data for your geographical zone.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <CardTitle>2. Dynamic Cost Generation & ML Inference</CardTitle>
                <p style={{ color: 'var(--color-text-secondary)', marginTop: '8px' }}>
                  XGBoost models predict the traffic delay and safety risk for every road segment. These are combined dynamically into a unified travel cost.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <CardTitle>3. A* Graph Search</CardTitle>
                <p style={{ color: 'var(--color-text-secondary)', marginTop: '8px' }}>
                  The engine calculates the optimal path between every stop using the predicted dynamic costs, rather than static distances.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <CardTitle>4. TSP Optimization</CardTitle>
                <p style={{ color: 'var(--color-text-secondary)', marginTop: '8px' }}>
                  Finally, it evaluates permutations of delivery orders to find the absolute most efficient sequence.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </PageContainer>
      <Footer />
    </>
  );
};
