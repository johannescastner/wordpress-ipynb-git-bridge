import Image from 'next/image';
import styles from '../styles/Home.module.css';

export default function Home() {
  return (
    <div>
      {/* Hero Section */}
      <section className={styles.hero}>
        <Image
          src="/images/human_tech_synergy.jpg"
          alt="Synergy between humans and technology"
          width={1920}
          height={1080}
          priority
        />
        <div className={styles.heroText}>
          <h1>Welcome to Towards People</h1>
          <p>Enhancing human creativity and purpose through ethical AI and collective intelligence.</p>
          <a href="#about" className={styles.ctaButton}>Learn More</a>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className={styles.about}>
        <div className={styles.textBlock}>
          <h2>About Towards People</h2>
          <p>
            We believe in the synergy of human creativity and AI capabilities.
            By adhering to ethical principles and fostering collective intelligence,
            we help organizations unlock their true potential.
          </p>
        </div>
        <div className={styles.imageBlock}>
          <Image
            src="/images/human_connection.jpg"
            alt="Human connection and collaboration"
            width={800}
            height={600}
          />
        </div>
      </section>

      {/* Principles Section */}
      <section className={styles.principles}>
        <h2>Our Principles</h2>
        <div className={styles.principlesContent}>
          <div className={styles.imageBlock}>
            <Image
              src="/images/personal_growth.jpg"
              alt="Personal growth through AI and leadership"
              width={800}
              height={600}
            />
          </div>
          <ul>
            <li>Synergy Between Human and AI Capabilities</li>
            <li>Human-Centered, Purpose-Driven Development</li>
            <li>Rigorous Experimentation</li>
            <li>Iterative Interaction for Continuous Improvement</li>
            <li>Leadership for Collective Intelligence</li>
          </ul>
        </div>
      </section>

      {/* Placeholder for Case Studies */}
      <section className={styles.caseStudies}>
        <h2>Case Studies</h2>
        <p>Explore how we’ve made a difference for organizations.</p>
        <Image
          src="/images/business_growth.jpg"
          alt="Business growth through AI solutions"
          width={1920}
          height={1080}
        />
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <h3>Contact Us</h3>
          <p>Stay connected and discover more about Towards People.</p>
        </div>
        <Image
          src="/images/communication.jpg"
          alt="Communication and connection"
          width={1920}
          height={600}
        />
      </footer>
    </div>
  );
}
