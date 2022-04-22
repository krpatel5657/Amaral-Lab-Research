import React from 'react';
import CardItem from './CardItem';
import './Cards.css'

function Cards() {
    return (
        <div className='cards'>
            <h1>Check out my projects:</h1>
            <div className='cards__container'>
                <div className='cards__wrapper'>
                    <ul className='cards__items'>
                        <CardItem
                            src="images/img-9.jpg"
                            text="Explore project number 1 here!"
                            label='Chocolate'
                            path='/projects'
                        />
                        <CardItem
                            src="images/img-2.jpg"
                            text="Explore project number 2 here!"
                            label='Fruits'
                            path='/projects'
                        />
                    </ul>
                    <ul className='cards__items'>
                        <CardItem 
                        src="images/img-8.jpg"
                        text="Explore project number 1 here!"
                        label='Chocolate'
                        path='/projects'
                        />
                        <CardItem
                            src="images/img-7.jpg"
                            text="Explore project number 2 here!"
                            label='Fruits'
                            path='/projects'
                        />
                        <CardItem
                            src="images/img-6.jpg"
                            text="Explore project number 3 here!"
                            label='Music'
                            path='/projects'
                        />
                    </ul>
                </div>
            </div>
        </div>
    );
}

export default Cards;
