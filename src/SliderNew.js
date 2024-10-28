import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HeadsetAttentionDisplay = () => {
  const [ranges, setRanges] = useState({
    high: 80,
    low: 50
  });
  const [liveAttention, setLiveAttention] = useState(0);
  const [position, setPosition] = useState({ x: 0, y: 0, rotation: 0 });

  useEffect(() => {
    // Fetch initial ranges
    axios.get('http://localhost:5000/get_ranges')
      .then(response => {
        setRanges(response.data);
      })
      .catch(error => {
        console.error('Error fetching ranges:', error);
      });

    // Set up polling for attention values
    const interval = setInterval(() => {
      axios.get('http://localhost:5000/get_attention')
        .then(response => {
          setLiveAttention(response.data.attention);
        })
        .catch(error => {
          console.error('Error fetching live attention:', error);
        });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const newPosition = getPosition(liveAttention);
    setPosition(newPosition);
  }, [liveAttention, ranges]);

  const handleSliderChange = (event) => {
    const { name, value } = event.target;
    const newRanges = {
      ...ranges,
      [name]: parseInt(value)
    };

    // Ensure high is always greater than low
    if (name === 'high' && parseInt(value) <= ranges.low) {
      newRanges.low = parseInt(value) - 1;
    } else if (name === 'low' && parseInt(value) >= ranges.high) {
      newRanges.high = parseInt(value) + 1;
    }

    setRanges(newRanges);

    axios.post('http://localhost:5000/update_ranges', newRanges)
      .then(response => {
        console.log('Ranges updated successfully');
      })
      .catch(error => {
        console.error('Error updating ranges:', error);
      });
  };

  const getPosition = (value) => {
    const containerSize = 64;
    const maxMove = containerSize / 4;

    if (value > ranges.high) {
      // A state - Move forward
      return { x: 0, y: -maxMove, rotation: 0 };
    } else {
      // D state - Stay/Stop
      return { x: 0, y: maxMove, rotation: 0 };
    }
  };

  const getActiveState = (value) => {
    if (value > ranges.high) return 'move';
    return 'stay';
  };

  const getWaveIntensity = (value) => {
    const intensity = value / 100;
    return Math.min(Math.max(intensity, 0), 1);
  };

  const getImageData = (value) => {
    const activeState = getActiveState(value);
    const intensity = getWaveIntensity(value);
    
    const images = {
      move: './Vector (1).png',  // Replace with your move state image
      stay: './vector (2).png'   // Replace with your stay state image
    };

    return {
      baseImage: images[activeState],
      opacity: intensity
    };
  };

  const activeState = getActiveState(liveAttention);
  const waveIntensity = getWaveIntensity(liveAttention);
  const imageData = getImageData(liveAttention);

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-2xl font-bold text-center">Attention Display</h2>
      
      <div className="text-center">
        <h3 className="text-lg font-semibold">Live Attention Value: {liveAttention}</h3>
        <p className="text-xl font-bold mt-2" style={{ 
          color: activeState === 'move' ? '#00ff00' : '#ff0000' 
        }}>
          {activeState === 'move' ? 'MOVE' : 'STAY'}
        </p>
      </div>

      <div className="relative w-64 h-64 mx-auto">
        {/* Headset container */}
        <div 
          className="absolute inset-0 flex items-center justify-center z-10 transition-all duration-300 ease-out"
          style={{
            transform: `translate(${position.x}px, ${position.y}px) rotate(${position.rotation}deg)`
          }}
        >
          <img 
            src={imageData.baseImage}
            alt="Headset State" 
            className="w-40 h-40 object-contain"
          />
        </div>
        
        {/* Wave container */}
        <div className="absolute inset-0">
          <div className={`wave ${activeState}-wave`} style={{
            animationDuration: `${2 / waveIntensity}s`,
            opacity: waveIntensity * 0.7
          }}></div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Move Threshold: {ranges.high}
          </label>
          <input
            type="range"
            name="high"
            min="0"
            max="100"
            value={ranges.high}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Stay Threshold: {ranges.low}
          </label>
          <input
            type="range"
            name="low"
            min="0"
            max="100"
            value={ranges.low}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">State Visualization</h3>
        <div className="w-full h-8 flex">
          {[...Array(100)].map((_, i) => (
            <div 
              key={i} 
              className={`w-1 h-full ${i + 1 > ranges.high ? 'move-bg' : 'stay-bg'}`}
            />
          ))}
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span>0</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100</span>
        </div>
      </div>

      <div className="mt-6 space-y-2">
        <h3 className="text-lg font-semibold">States:</h3>
        <p className="text-green-600">MOVE (A): value > {ranges.high}</p>
        <p className="text-red-600">STAY (D): value ≤ {ranges.high}</p>
      </div>

      <style jsx>{`
        @keyframes singlePulse {
          0% { transform: scale(0); opacity: 0.7; }
          100% { transform: scale(1); opacity: 0; }
        }

        .wave {
          position: absolute;
          animation: singlePulse 2s ease-out infinite;
        }

        .move-wave {
          top: 0px;
          left: 0%;
          right: 0%;
          height: 60px;
          border-radius: 50% 50% 0 0;
          background-color: rgba(0, 255, 0, 0.5);
        }

        .stay-wave {
          bottom: 0px;
          left: 0%;
          right: 0%;
          height: 60px;
          border-radius: 0 0 50% 50%;
          background-color: rgba(255, 0, 0, 0.5);
        }

        .move-bg { background-color: #00ff00; }
        .stay-bg { background-color: #ff0000; }
      `}</style>
    </div>
  );
};

export default HeadsetAttentionDisplay;