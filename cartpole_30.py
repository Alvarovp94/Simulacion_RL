"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.vector import VectorEnv
from gymnasium.vector.utils import batch_space

import pymunk
import pygame

import pymunk.pygame_util

import time



class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
       
    metadata = {
        "render_modes": ["human"],
        "render_fps": 60,
    }

    def __init__(self, max_episode_steps: int = 300, render_mode: Optional[str] = None,):
        
        print("####### estamos trabajando #######")
        
        self.screen = None
        self.clock = None
        self.pymunk = None
        
        
        self.screen_width = 600
        self.screen_high = 400
        

        self.init_pygame()
        self.init_pymunk()
        

        ####### Cuerpos estaticos ###########     
        
        dist_ref = 5
        
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        static_body.position = (self.screen_width/2, dist_ref)
        static_line = pymunk.Segment(static_body, (-50, 0), (50, 0), 5)
        self.space.add(static_body, static_line)

        
        static_body_2 = pymunk.Body(body_type=pymunk.Body.STATIC)
        static_body_2.position = (self.screen_width/2, self.screen_high - dist_ref)
        static_line_2 = pymunk.Segment(static_body_2, (-50, 0), (50, 0), 5)
        self.space.add(static_body_2, static_line_2)
        
        
        ########### Caja de fusibles ###################
        
        # Cuerpo dinámico que estará sujeto al resorte (ahora un rectángulo)
        zoom = 10
        
        mass = 0.1
        width = 6*zoom
        height = 2*zoom
        moment = float('inf')
        self.body = pymunk.Body(mass, moment)
        
        self.body_position_init = 300, dist_ref + 50 + height/2
        
        self.body.position = self.body_position_init
        # Definimos el rectángulo como un polígono. Los puntos se definen respecto al centro del cuerpo.
        rectangle = pymunk.Poly.create_box(self.body, (width, height))
        self.space.add(self.body, rectangle)
        # Crear el resorte

        k = 10 # Nueva constante del resorte (stiffness)
        c = 1  # Nueva constante de amortiguación (damping)
        L1 = self.body.position.y - static_body.position.y
        print("L1 de resorte es: ", L1)
        spring = pymunk.DampedSpring(self.body, static_body, (0,0), (0,0), L1, k, c)
        self.space.add(spring)



        # Restricción de movimiento en el eje X usando un GrooveJoint
        groove = pymunk.GrooveJoint(static_body, self.body, (0, -10), (0, 700), (0, 0))
        self.space.add(groove)


        mass_top = 0.1 # Masa del nuevo rectángulo
        width_top = 2*zoom  # Ancho
        height_top = 6.5*zoom  # Altura
        self.hub = height_top
        moment_top = float('inf')
        print("Todo el hub es: ", self.hub)


        b = 0.8

        self.top_body = pymunk.Body(mass_top, moment_top)     
        top_rectangle = pymunk.Poly.create_box(self.top_body, (width_top, height_top))
        top_rectangle.friction = b
        self.top_body_position_init = self.body.position.x + width/2 -  width_top/2 , self.body.position.y + (height/2 + height_top/2)
        self.top_body.position = self.top_body_position_init
        self.space.add(self.top_body, top_rectangle)


        self.top_body_2 = pymunk.Body(mass_top, moment_top)
        top_rectangle_2 = pymunk.Poly.create_box(self.top_body_2, (width_top, height_top))
        top_rectangle_2.friction = b
        top_rectangle_2.collision_type = 4
        self.top_body_2_position_init = self.body.position.x - width/2 +  width_top/2 , self.body.position.y + (height/2 + height_top/2)
        self.top_body_2.position = self.top_body_2_position_init
        self.space.add(self.top_body_2, top_rectangle_2) 
        
        self.point_m1 = self.top_body_2.position.y + height_top/2
        self.point_m2 = self.top_body_2.position.y - height_top/2
        self.end_hub = self.body.position.y + height/2
        
        print("el punto m1 es:", self.point_m1)
        print("el punto m2 es:", self.point_m2)
        print("el punto m3 es:", self.body.position.y + height/2)
        
        
        
        ####################
        # Restricción de movimiento en el eje X usando un GrooveJoint
        groove = pymunk.GrooveJoint(static_body, self.top_body, (width/2 - width_top/2, -1000), (width/2 - width_top/2, 1000), (0, 0))
        self.space.add(groove)
        
        joint_primero = pymunk.PinJoint(self.body, self.top_body, (width/2 - width_top/2, 0), (0,0))
        self.space.add(joint_primero)

        #groove = pymunk.GrooveJoint(static_body, self.top_body, (width/2 - width_top/2, -1000), (width/2 - width_top/2, 1000), (0, -height_top/2))
        #self.space.add(groove)

        # Restricción de movimiento en el eje X usando un GrooveJoint
        groove = pymunk.GrooveJoint(static_body, self.top_body_2, (-width/2 + width_top/2, -1000), (-width/2 + width_top/2, 1000), (0, 0))
        self.space.add(groove)
        
        joint_segundo = pymunk.PinJoint(self.body, self.top_body_2, (-width/2 + width_top/2, 0), (0,0))
        self.space.add(joint_segundo)

        #groove = pymunk.GrooveJoint(static_body, self.top_body_2, (-width/2 + width_top/2, -1000), (-width/2 + width_top/2, 1000), (0, -height_top/2))
        #self.space.add(groove)
        
        self.zona_final = 0
        
        #print("self.body.position###### 2 ", self.body.position)

          # Posicionándolo justo encima del rectángulo existente
        
        #########################################################################
        
        ############robot #################


        # Cuerpo dinámico que estará sujeto al resorte (ahora un rectángulo)
        mass_2 = 0.01
        width_2 = 3*zoom
        height_2 = 3*zoom
        moment_2 = float('inf')
        self.body_2 = pymunk.Body(mass_2, moment_2)
        
        self.body_2_position_init = 300, self.screen_high - dist_ref - 35
        
        self.body_2.position = self.body_2_position_init
        rectangle_2 = pymunk.Poly.create_box(self.body_2, (width_2, height_2))
        #rectangle_2.collision_type = 2
        self.space.add(self.body_2, rectangle_2)
        

        '''
        mass_3 = 0.001
        width_3 = 3*zoom
        height_3 = 1*zoom
        moment_3 = float('inf')
        self.body_3 = pymunk.Body(mass_3, moment_3)
        
        self.body_3_position_init = 300, self.body_2.position.y - height_3/2 - height_2/2 - 20

        self.body_3.position = self.body_3_position_init
        rectangle_3 = pymunk.Poly.create_box(self.body_3, (width_3, height_3))
        #rectangle_3.col
        self.space.add(self.body_3, rectangle_3)
        '''

        '''
        k_sen = 2000 # Nueva constante del resorte (stiffness)
        c_sen = 100  # Nueva constante de amortiguación (damping)
        L1_sen = self.body_2.position.y - self.body_3.position.y 
        print("este es L1 del sensor ########", L1_sen)
        spring_sen = pymunk.DampedSpring(self.body_2, self.body_3, (0,0), (0,0), L1_sen, k_sen, c_sen)
        self.space.add(spring_sen)
        '''
        
        groove = pymunk.GrooveJoint(static_body, self.body_2, (0, -1000), (0, 1000), (0, 0))
        self.space.add(groove)
        
        #groove = pymunk.GrooveJoint(static_body, self.body_3, (0, -1000), (0, 1000), (0, 0))
        #self.space.add(groove)
        
        
        ##################### Fusible #############################


        mass_insert = 0.001  # Masa del nuevo rectángulo
        width_insert = 2.025*zoom  # Ancho
        self.height_insert = 7*zoom # Altura
        moment_insert = float('inf')


        b1 = 0.8

        self.insert_body = pymunk.Body(mass_insert, moment_insert)
        
        self.insert_body_position_init = self.body.position.x , self.body_2.position.y - height_2/2 - self.height_insert/2

        self.insert_body.position =  self.insert_body_position_init

        insert_rectangle = pymunk.Poly.create_box(self.insert_body, (width_insert, self.height_insert))
        insert_rectangle.friction = b1
        insert_rectangle.collision_type = 1
        self.space.add(self.insert_body, insert_rectangle)
        
        self.a_point = self.insert_body.position.y - self.height_insert/2
        print("el punta a esta en ", self.a_point)
        


        # Restricción de movimiento en el eje X usando un GrooveJoint
        groove = pymunk.GrooveJoint(static_body, self.insert_body, (0, -1000), (0, 1000), (0, 0))
        self.space.add(groove)


        self.joint_greifer = pymunk.PinJoint(self.body_2,self.insert_body, (0, 0), (0, 0))
        self.space.add(self.joint_greifer)

   
        
        ###########################################################
        
        self.robot_greifer = True
        
        self.max_episode_steps = max_episode_steps
        
        self.dist = 1000

        #################
        
        self.current_time_step = 0
        
        self.val_terminated = False
        self.val_truncated = False
        self.reward = 0
        
        self.first_step = False
        
        self.contrar_premio = 0
        
        
        ###########
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([600, 600]), dtype=np.float32)
        
    def reset_position(self):
        self.body.position = self.body_position_init   
        self.body.velocity = 0, 0
        self.body.angular_velocity = 0
        
        self.top_body.position = self.top_body_position_init
        self.top_body.velocity = 0, 0
        self.top_body.angular_velocity = 0
        
        self.top_body_2.position = self.top_body_2_position_init
        self.top_body_2.velocity = 0, 0
        self.top_body_2.angular_velocity = 0
        
        self.body_2.position = self.body_2_position_init
        self.body_2.velocity = 0, 0
        self.body_2.angular_velocity = 0
        
        #self.body_3.position = self.body_3_position_init
        #self.body_3.velocity = 0, 0 
        #self.body_3.angular_velocity = 0
        
        self.insert_body.position =  self.insert_body_position_init
        self.insert_body.velocity = 0, 0
        self.insert_body.angular_velocity = 0
        
        self.contrar_premio = 0
        
        

        

        

        
    def init_pygame(self):
        
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_high))
            

        if self.clock is None:
            self.clock = pygame.time.Clock()
        



    def init_pymunk(self):
        
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        #self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        pymunk.pygame_util.positive_y_is_up = True
        
        
        
        
        

    def step(self, action):
              

        #print("la velocidad es", action)
        
        self.sprint_last_step = False
        

        self.body_2.velocity = 0, action*-50
        
        
        self.space.step(1/60)
        
        
        mesure_point = self.insert_body.position.y - self.hub/2
        
        lim_mesure_point = self.insert_body.position.x
        
        
        pos_state = 1
           
        self.dist = (self.insert_body.position.y - self.hub/2) - self.end_hub
        
        #print(" ########### Esto es dist: ", dist)
        
        self.reward = self.reward_func(mesure_point, self.dist, pos_state);
        
        #print("valor del premio #####", self.reward)
        

        
        self.state =  [mesure_point, 450]
        
        
        self.current_time_step +=1
        
        self.val_terminated = False
        self.val_truncated = False
        self.limite_max_episodio = False
        
        
        #print("valor de reinicio #####", self.current_time_step >= self.max_episode_steps , "es por: ", self.current_time_step)
        if self.current_time_step >= self.max_episode_steps:
            self.val_terminated = False
            self.val_truncated = True
            self.limite_max_episodio = True

            print("fue truncado #####")
            
        #Restricciones ventana de simulacion ################

        if self.body_2.position.y < 0 or self.body_2.position.y > 400:
            self.val_terminated = False
            self.val_truncated = True
            #self.sprint_last_step = True
            print("fue truncado #####")
            
        '''
        if self.body_3.position.y < 0 or self.body_3.position.y > 400:
            self.val_truncated = True
            self.val_terminated = True
            self.sprint_last_step = True
            print("fue truncado #####")
        '''
            
        if self.insert_body.position.y < 0 or self.insert_body.position.y > 400:
            self.val_terminated = False
            self.val_truncated = True
            #self.sprint_last_step = True
            print("fue truncado #####")
            
        ####################################################
        
        #Restriccion fiscas de seguridad #################
        
        if self.insert_body.position.x > 310 or self.insert_body.position.x < 290:
            self.val_terminated = False
            self.val_truncated = True
            #self.sprint_last_step = True
            print("fue truncado por salirse del lugar el fusible ############ ", self.insert_body.position.x)
            time.sleep(5)
            
            pos_state = 3

            self.reward = self.reward_func(mesure_point, self.dist, pos_state)
            
            
        if self.body_2.position.x > 310 or self.body_2.position.x < 290:
            self.val_terminated = False
            self.val_truncated = True
            #self.sprint_last_step = True
            print("fue truncado por salirse del lugar el robot ############ ", self.body_2.position.x)
            time.sleep(5)
            
            pos_state = 3

            self.reward = self.reward_func(mesure_point, self.dist, pos_state)
            
        '''       
        if self.body_3.position.x > 310 or self.body_3.position.x < 290:
            self.val_truncated = True
            self.val_terminated = True
            self.sprint_last_step = True
            print("fue truncado por salirse del lugar la pinza ############ ", self.body_3.position.x)
            time.sleep(5)
            
            state = 3

            self.reward = self.reward_func(mesure_point, self.dist, state)
        '''
            
        if  self.insert_body.position.y > self.body_2.position.y:
            
            self.val_terminated = False
            self.val_truncated = True
            #self.sprint_last_step = True
            print("fue truncado por roper el orden logico ########### ")
            time.sleep(5)
            
            pos_state = 3

            self.reward = self.reward_func(mesure_point, self.dist, pos_state)
        ###########################################################
        
        
            
        '''
        if self.dist == 0:
            self.val_truncated = True
            self.val_terminated = True
            print("fue truncado #####")
        '''
                  
            
        self.render()
            
        
        if self.limite_max_episodio == True and self.dist < self.hub and self.dist > 0 :

            
            self.robot_greifer = True
            print("dejamos el fusibleeeeeeeeeeeee")

            for i in range(1000): 
                if self.robot_greifer == True:

                    
                    self.space.remove(self.joint_greifer)
                    self.robot_greifer = False
                    
                    self.space.step(1/60)
                    self.render()

                if self.body_2.position.y < self.body_2_position_init[1]:
                    #print("########## reversa ######################")
                    self.body_2.velocity = 0, 40
                    #self.body_3.velocity = 0, 25
                    
                if self.body_2.position.y >= self.body_2_position_init[1]:
                    #print("########## reversa ######################")  
                    self.body_2.force = 0, 0
                    self.body_2.velocity = 0, 0
                    #self.body_3.force = 0, 0
                    #self.body_3.velocity = 0, 0
                    self.body_2.angular_velocity = 0
                    #self.body_3.angular_velocity = 0
                    
                    
                    break
                
  
                self.space.step(1/60)
                    
                self.render()
                
                
            pos_state = 2
            
            self.reward = self.reward_func(mesure_point, self.dist, pos_state)
            
            if self.reward[1] == True:
                
                self.val_terminated = True
                self.val_truncated = False
                
                
            print("############## Acabo la magia ###############")
    
        return np.array(self.state, dtype=np.float32), self.reward[0], self.val_terminated, self.val_truncated, {}
    
    
    def reward_func(self, point, dist, state):
        
        self.point_a = self.a_point
        self.point_b = self.point_m1
        self.largo = self.hub
        
        self.target = 20
        self.p1 = self.target  + 6
        self.p2 = self.target  - 0
        
        self.screen_high = 400

        self.reward_max_1 = 10
        self.reward_max_2 = 4000
        
        acumulado = 0
        objetivo = False
        
        #print(" ########### state es:", state)
        #print("##### point es:", point)
        #print("##### point_b es:", self.point_b)

        #reward = 777
        
        if state == 1:
            
            if point >= self.point_a:
                reward = -self.reward_max_1*(point - self.point_a)/(self.screen_high - self.point_a)
                self.zona_final = 1
                #print("se acabo atras la oepracion ########")
                #print("posicion de punto:",point, "posicion de marca", self.point_a,"el premio es", reward)
            if point < self.point_a and point >= (self.point_b  - 20):
                reward = (self.reward_max_1)*(point - self.point_a)/( self.point_b - self.point_a)
                self.zona_final = 1
                #print("se acabo delante ########")
                #print("posicion de punto:",point, "posicion de marca", self.point_a,"el premio es", reward)
                
            if point < (self.point_b - 20) and point >= 10:
                reward = self.reward_max_1
                self.zona_final = 1
                #print("se acabo delante ########")
                #print("posicion de punto:",point, "posicion de marca", self.point_a,"el premio es", reward)
              
            acumulado = reward
                        
        if state == 2:
            
            if dist >= self.target:
                reward_2 = self.reward_max_2*(dist - self.hub)/(self.target - self.hub - self.p1)
                self.zona_final = 2
                print("aun no llegamos al target:", dist, "el premio es", reward_2 )
                
            if dist < self.target:
                reward_2 = self.reward_max_2*dist/self.target - 4000
                self.zona_final = 4
                print("nos pasamos del target:", dist, "el premio es", reward_2 )
                
            
            if dist < 26 and dist > 20 :
                objetivo = True
                print("ESTAMOS EN LA ZONA", dist, "el premio es", reward_2 )
                

                
            acumulado = self.reward_max_1 + reward_2 + 100
            
            if dist == 0:
                acumulado = -1000
                
        if state == 3:
            
            acumulado = -2000
            
        self.contar(acumulado)
                

        return acumulado , objetivo
    
    def contar(self, premio):
        
        self.contrar_premio = self.contrar_premio + premio
        
        return self.contrar_premio
                
                

        

    def reset(self, seed=None, options=None):
        
        #### reinit ######################
        
        print("La posicion global fue: ", self.insert_body.position.y - self.hub/2) 
        print("el premio acumulado fue:", self.contrar_premio)
        

        super().reset(seed=seed)
        
        self.reset_position()
        

        if self.joint_greifer not in self.space.constraints:
            self.space.add(self.joint_greifer)
            #print("no habia join, lo repusimos")
            
        self.robot_greifer == True

        
        print("el premio fue: ", self.reward)
        print("la dist final fue: ", self.dist)
        print("la zona final fue:", self.zona_final)
        
        self.dist = 1000
        self.zona_final = 0


        #self.body.position = (300 , 10 + 5/2 + 20)

        
        self.first_step = False
        #self.space.step(1/60)
        
        self.val_terminated = False
        self.val_truncated = False
        self.reward = 0
        

        

        mesure_point = self.insert_body.position.y - self.height_insert/2
        
        self.current_time_step = 0

        ######
        self.state =  [mesure_point, 450]
        
        self.space.step(1/60)
        self.render()
        

        return np.array(self.state, dtype=np.float32), {}
    
        

    def render(self):
        
            
       #self.screen = pygame.display.set_mode((600, 400))
        
        # Limpiar la pantalla
        self.screen.fill((255, 255, 255))
        
        # Dibujar los objetos de pymunk
        self.space.debug_draw(self.draw_options)
        
        
        pygame.event.pump()
        # Esperar un poco para el siguiente cuadro
        self.clock.tick(60)
        
        #pygame.draw.line(self.screen, self.RED, (600 * 0.75, 400), (600 * 0.75, 350), 5)
        
        pygame.display.flip()
        
        
        '''
        if self.first_step == False:
            self.first_step = True
            time.sleep(3)
            print("Este es la pausa del primer estado")
        '''
        

    def close(self):
        pygame.display.quit()
        pygame.quit()





